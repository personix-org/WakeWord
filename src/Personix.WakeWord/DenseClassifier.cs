using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;

namespace Personix.WakeWord;

/// <summary>
/// A wake word classifier — the trained head that sits on top of a window of speech embeddings.
///
/// The openWakeWord chain is three models: melspectrogram, embedding and classifier. The first two
/// are pre-trained files that never change, so ONNX Runtime executes them. Only the third one is
/// trained, and it is small enough — three linear layers, 1536 → 128 → 64 → 1 — that the forward
/// pass fits in a handful of loops.
///
/// The format exists so that training and inference can live in one language: a trainer written
/// with TorchSharp writes the file and the detector reads it. TorchSharp cannot export to ONNX,
/// so keeping the head in ONNX would mean keeping the trainer in Python.
///
/// Weights of a layer lie contiguously in memory and the dot product runs on SIMD, because in
/// production the classifier scores every 80 ms frame and each extra wake word adds a pass.
///
/// One instance belongs to one thread — layer results are held in buffers shared between calls
/// to <see cref="Score"/>.
/// </summary>
public sealed class DenseClassifier
{
    private const string Magic = "WWCLS1";

    /// <summary>Weights of a layer, row after row: <c>[output * inputs + input]</c>.</summary>
    private readonly float[][] _weights;
    private readonly float[][] _biases;
    private readonly int[] _inputs;

    /// <summary>Layer results. They swap, so the input of a layer survives writing its output.</summary>
    private readonly float[] _frontBuffer;
    private readonly float[] _backBuffer;

    /// <param name="layers">Layers from input to output. The last one must have a single output.</param>
    public DenseClassifier(IReadOnlyList<DenseLayer> layers)
    {
        var validated = Validated(layers);

        _weights = new float[validated.Length][];
        _biases = new float[validated.Length][];
        _inputs = new int[validated.Length];

        var widest = 0;

        for (var l = 0; l < validated.Length; l++)
        {
            var layer = validated[l];
            var inputs = layer.Weights[0].Length;
            var outputs = layer.Biases.Length;

            var flat = new float[outputs * inputs];
            for (var o = 0; o < outputs; o++)
            {
                    layer.Weights[o].CopyTo(flat, o * inputs);
            }

            _weights[l] = flat;
            _biases[l] = layer.Biases;
            _inputs[l] = inputs;

            widest = Math.Max(widest, Math.Max(inputs, outputs));
        }

        _frontBuffer = new float[widest];
        _backBuffer = new float[widest];
    }

    /// <summary>Values expected on the input — for openWakeWord 16 frames of 96, so 1536.</summary>
    public int InputSize => _inputs[0];

    /// <summary>
    /// Probability from 0 to 1 that the window holds the wake word. ReLU sits between layers and
    /// a sigmoid closes the last one — the same shape as the trained network.
    /// </summary>
    public float Score(ReadOnlySpan<float> window)
    {
        if (window.Length != InputSize)
        {
                throw new ArgumentException($"Expected {InputSize} values, got {window.Length}.", nameof(window));
        }

        var current = window;
        Span<float> next = _frontBuffer;
        Span<float> spare = _backBuffer;

        for (var l = 0; l < _weights.Length; l++)
        {
            var weights = _weights[l].AsSpan();
            var biases = _biases[l];
            var inputs = _inputs[l];
            var output = next[..biases.Length];
            var lastLayer = l == _weights.Length - 1;

            // Four rows at a time: the input is loaded once per step and feeds four independent
            // accumulators, which keeps the multiply-add units busy instead of waiting on one chain.
            var o = 0;
            for (; o + 4 <= biases.Length; o += 4)
            {
                DotFourRows(weights.Slice(o * inputs, 4 * inputs), inputs, current, output.Slice(o, 4));

                for (var r = 0; r < 4; r++)
                {
                    output[o + r] += biases[o + r];
                }
            }

            for (; o < biases.Length; o++)
            {
                output[o] = biases[o] + Dot(weights.Slice(o * inputs, inputs), current);
            }

            // ReLU between layers, the last one is closed by the sigmoid below.
            if (!lastLayer)
            {
                for (var i = 0; i < output.Length; i++)
                {
                    output[i] = Math.Max(0f, output[i]);
                }
            }

            current = output;

            var used = next;
            next = spare;
            spare = used;
        }

        return Sigmoid(current[0]);
    }

    /// <summary>Writes the classifier in the <c>.wwc</c> format that <see cref="Load"/> reads.</summary>
    public void Save(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        writer.Write(Encoding.ASCII.GetBytes(Magic));
        writer.Write(_weights.Length);

        for (var l = 0; l < _weights.Length; l++)
        {
            writer.Write(_inputs[l]);           // inputs
            writer.Write(_biases[l].Length);    // outputs

            foreach (var weight in _weights[l])
            {
                    writer.Write(weight);
            }

            foreach (var bias in _biases[l])
            {
                    writer.Write(bias);
            }
        }
    }

    /// <summary>Reads a classifier written by <see cref="Save"/>.</summary>
    /// <exception cref="InvalidDataException">The file is not a classifier, or is truncated.</exception>
    public static DenseClassifier Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        try
        {
            var magic = Encoding.ASCII.GetString(reader.ReadBytes(Magic.Length));
            if (magic != Magic)
            {
                throw new InvalidDataException(
                    $"{path} is not a wake word classifier (expected {Magic}, found '{magic}')");
            }

            var layerCount = reader.ReadInt32();
            if (layerCount is < 1 or > 16)
            {
                    throw new InvalidDataException($"{path} claims {layerCount} layers, which makes no sense");
            }

            var layers = new DenseLayer[layerCount];

            for (var l = 0; l < layerCount; l++)
            {
                var inputs = reader.ReadInt32();
                var outputs = reader.ReadInt32();

                var weights = new float[outputs][];
                for (var o = 0; o < outputs; o++)
                {
                    weights[o] = new float[inputs];
                    for (var i = 0; i < inputs; i++)
                    {
                            weights[o][i] = reader.ReadSingle();
                    }
                }

                var biases = new float[outputs];
                for (var o = 0; o < outputs; o++)
                {
                        biases[o] = reader.ReadSingle();
                }

                layers[l] = new DenseLayer(weights, biases);
            }

            return new DenseClassifier(layers);
        }
        catch (EndOfStreamException e)
        {
            throw new InvalidDataException($"{path} ends earlier than it should — the file is truncated", e);
        }
        catch (ArgumentException e)
        {
            throw new InvalidDataException($"{path} carries nonsensical dimensions", e);
        }
    }

    /// <summary>Dot products of four consecutive weight rows with the same input.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DotFourRows(ReadOnlySpan<float> rows, int inputs, ReadOnlySpan<float> values, Span<float> results)
    {
        var width = Vector<float>.Count;
        var row0 = rows[..inputs];
        var row1 = rows.Slice(inputs, inputs);
        var row2 = rows.Slice(2 * inputs, inputs);
        var row3 = rows.Slice(3 * inputs, inputs);

        Vector<float> sum0 = default;
        Vector<float> sum1 = default;
        Vector<float> sum2 = default;
        Vector<float> sum3 = default;
        var i = 0;

        for (; i <= inputs - width; i += width)
        {
            var value = new Vector<float>(values.Slice(i, width));
            sum0 += new Vector<float>(row0.Slice(i, width)) * value;
            sum1 += new Vector<float>(row1.Slice(i, width)) * value;
            sum2 += new Vector<float>(row2.Slice(i, width)) * value;
            sum3 += new Vector<float>(row3.Slice(i, width)) * value;
        }

        results[0] = Vector.Sum(sum0);
        results[1] = Vector.Sum(sum1);
        results[2] = Vector.Sum(sum2);
        results[3] = Vector.Sum(sum3);

        // Tail shorter than one vector.
        for (; i < inputs; i++)
        {
            results[0] += row0[i] * values[i];
            results[1] += row1[i] * values[i];
            results[2] += row2[i] * values[i];
            results[3] += row3[i] * values[i];
        }
    }

    /// <summary>Dot product of one weight row with the layer input.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Dot(ReadOnlySpan<float> weights, ReadOnlySpan<float> values)
    {
        var width = Vector<float>.Count;
        var accumulator = Vector<float>.Zero;
        var i = 0;

        for (; i <= weights.Length - width; i += width)
        {
                accumulator += new Vector<float>(weights.Slice(i, width)) * new Vector<float>(values.Slice(i, width));
        }

        var sum = Vector.Sum(accumulator);

        // Tail shorter than one vector.
        for (; i < weights.Length; i++)
        {
                sum += weights[i] * values[i];
        }

        return sum;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static DenseLayer[] Validated(IReadOnlyList<DenseLayer> layers)
    {
        if (layers.Count == 0)
        {
                throw new ArgumentException("A classifier needs at least one layer.", nameof(layers));
        }

        foreach (var layer in layers)
        {
            if (layer.Weights.Length != layer.Biases.Length)
            {
                throw new ArgumentException(
                    $"A layer has {layer.Weights.Length} weight rows but {layer.Biases.Length} biases.",
                    nameof(layers));
            }

            if (layer.Weights.Length == 0)
            {
                    throw new ArgumentException("A layer with no outputs.", nameof(layers));
            }
        }

        if (layers[^1].Biases.Length != 1)
        {
            throw new ArgumentException(
                $"The last layer must have one output, it has {layers[^1].Biases.Length}.", nameof(layers));
        }

        return [.. layers];
    }
}
