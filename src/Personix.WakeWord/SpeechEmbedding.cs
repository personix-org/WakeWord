using Microsoft.ML.OnnxRuntime;

namespace Personix.WakeWord;

/// <summary>
/// The speech embedding model, run incrementally.
///
/// The model reads a window of 76 mel frames and answers with 96 numbers. In a stream the window
/// moves by 8 frames a step, so nine tenths of it are the same frames as last time — and because
/// every convolution in the network is "valid" along time, the activations for those frames are
/// the same too. They are kept between steps, and each step computes only the columns the new
/// frames touch.
///
/// For that the network is split into six blocks at its pooling layers. Every block is its own
/// ONNX graph with a free time dimension: a full window through it fills the cache, a few columns
/// through it advance the cache. The weights are untouched, and the result is the same function
/// as one pass through the whole model.
///
/// One instance belongs to one thread.
/// </summary>
internal sealed class SpeechEmbedding : IDisposable
{
    /// <summary>Mel frames in a full window.</summary>
    public const int WindowFrames = 76;

    /// <summary>Mel frames a step moves the window by.</summary>
    public const int HopFrames = 8;

    /// <summary>Mel bands per frame.</summary>
    public const int MelBins = 32;

    /// <summary>Numbers in one embedding.</summary>
    public const int Size = 96;

    /// <summary>Mel frames a step needs — the new ones plus the reach of block 0's two time convolutions.</summary>
    public const int StepFrames = HopFrames + 4;

    /// <summary>
    /// One block of the network. The cache holds its output for a full window: <c>Channels</c> planes
    /// of <c>Time</c> columns by <c>Frequency</c> rows. A step feeds the block the last
    /// <c>StepInput</c> columns of the previous cache and gets <c>StepNew</c> new columns back.
    /// </summary>
    private readonly record struct Block(int Channels, int Time, int Frequency, int StepInput, int StepNew);

    private static readonly Block[] Layout =
    [
        new(Channels: 1, Time: WindowFrames, Frequency: MelBins, StepInput: StepFrames, StepNew: HopFrames),
        new(Channels: 24, Time: 36, Frequency: 16, StepInput: 8, StepNew: 4),
        new(Channels: 48, Time: 32, Frequency: 8, StepInput: 8, StepNew: 4),
        new(Channels: 72, Time: 14, Frequency: 4, StepInput: 6, StepNew: 2),
        new(Channels: 96, Time: 10, Frequency: 2, StepInput: 6, StepNew: 2),
        new(Channels: 96, Time: 3, Frequency: 1, StepInput: 3, StepNew: 1),
    ];

    private const int BlockCount = 6;
    private readonly string[] _inputNames = ["input"];
    private readonly string[] _outputNames = ["output"];
    private readonly RunOptions _runOptions = new();

    private readonly InferenceSession[] _sessions = new InferenceSession[BlockCount];

    /// <summary>Output of block i for the whole window, laid out <c>[channel][time][frequency]</c>. Block 0's "output" is the mel window itself.</summary>
    private readonly float[][] _cache = new float[BlockCount][];

    private readonly float[][] _stepInput = new float[BlockCount][];
    private readonly float[][] _stepOutput = new float[BlockCount][];
    private readonly OrtValue[][] _stepInputs = new OrtValue[BlockCount][];
    private readonly OrtValue[][] _stepOutputs = new OrtValue[BlockCount][];

    private readonly float[] _embedding = new float[Size];

    /// <param name="modelDirectory">Directory holding <c>embedding_b0.onnx</c> … <c>embedding_b5.onnx</c>.</param>
    /// <param name="options">Session settings — see <see cref="WakeWordFeatures.SequentialOptions"/>.</param>
    public SpeechEmbedding(string modelDirectory, SessionOptions options)
    {
        for (var b = 0; b < BlockCount; b++)
        {
            var block = Layout[b];
            var last = b + 1 == BlockCount;

            _sessions[b] = new InferenceSession(Path.Combine(modelDirectory, $"embedding_b{b}.onnx"), options);
            _cache[b] = new float[block.Channels * block.Time * block.Frequency];

            _stepInput[b] = new float[block.Channels * block.StepInput * block.Frequency];
            _stepInputs[b] = [OrtValue.CreateTensorValueFromMemory(_stepInput[b], [1, block.Channels, block.StepInput, block.Frequency])];

            // The last block ends with the model's own reshape to [1, 1, 1, 96]; the others hand
            // over new columns of the next block's cache.
            if (last)
            {
                _stepOutput[b] = new float[Size];
                _stepOutputs[b] = [OrtValue.CreateTensorValueFromMemory(_stepOutput[b], [1, 1, 1, Size])];
            }
            else
            {
                var next = Layout[b + 1];
                _stepOutput[b] = new float[next.Channels * next.StepNew * next.Frequency];
                _stepOutputs[b] = [OrtValue.CreateTensorValueFromMemory(_stepOutput[b], [1, next.Channels, next.StepNew, next.Frequency])];
            }
        }
    }

    /// <summary>
    /// Fills the cache from a full window — one pass of every block over all of it. The window
    /// the reference implementation starts from is all ones.
    /// </summary>
    /// <param name="window"><see cref="WindowFrames"/> frames of <see cref="MelBins"/>, frame after frame.</param>
    public void Reset(ReadOnlySpan<float> window)
    {
        if (window.Length != WindowFrames * MelBins)
        {
            throw new ArgumentException($"Expected {WindowFrames * MelBins} values, got {window.Length}.", nameof(window));
        }

        // A mel frame is [time][frequency], which is exactly block 0's single-channel plane.
        window.CopyTo(_cache[0]);

        // Full passes are rare — on start and on reset — so their tensors are made on the spot.
        for (var b = 0; b + 1 < BlockCount; b++)
        {
            var block = Layout[b];
            var next = Layout[b + 1];

            using var input = OrtValue.CreateTensorValueFromMemory(_cache[b], [1, block.Channels, block.Time, block.Frequency]);
            using var output = OrtValue.CreateTensorValueFromMemory(_cache[b + 1], [1, next.Channels, next.Time, next.Frequency]);
            _sessions[b].Run(_runOptions, _inputNames, [input], _outputNames, [output]);
        }
    }

    /// <summary>
    /// Moves the window by <see cref="HopFrames"/> frames and returns the embedding of the new window.
    /// </summary>
    /// <param name="frames">
    /// The last <see cref="StepFrames"/> mel frames of the new window, frame after frame — the
    /// <see cref="HopFrames"/> new ones preceded by the four before them.
    /// </param>
    public ReadOnlySpan<float> Advance(ReadOnlySpan<float> frames)
    {
        if (frames.Length != StepFrames * MelBins)
        {
            throw new ArgumentException($"Expected {StepFrames * MelBins} values, got {frames.Length}.", nameof(frames));
        }

        frames.CopyTo(_stepInput[0]);

        for (var b = 0; b + 1 < BlockCount; b++)
        {
            var next = Layout[b + 1];

            _sessions[b].Run(_runOptions, _inputNames, _stepInputs[b], _outputNames, _stepOutputs[b]);

            Append(_cache[b + 1], next, _stepOutput[b]);
            TakeLast(_cache[b + 1], next, _stepInput[b + 1]);
        }

        var last = BlockCount - 1;
        _sessions[last].Run(_runOptions, _inputNames, _stepInputs[last], _outputNames, _stepOutputs[last]);
        _stepOutput[last].CopyTo(_embedding);

        return _embedding;
    }

    /// <summary>Drops the oldest <c>StepNew</c> columns of a cache and appends the new ones.</summary>
    private static void Append(float[] cache, Block block, float[] columns)
    {
        var plane = block.Time * block.Frequency;
        var keep = (block.Time - block.StepNew) * block.Frequency;
        var fresh = block.StepNew * block.Frequency;

        for (var c = 0; c < block.Channels; c++)
        {
            Array.Copy(cache, (c * plane) + fresh, cache, c * plane, keep);
            Array.Copy(columns, c * fresh, cache, (c * plane) + keep, fresh);
        }
    }

    /// <summary>Copies the last <c>StepInput</c> columns of every channel into a step input.</summary>
    private static void TakeLast(float[] cache, Block block, float[] destination)
    {
        var plane = block.Time * block.Frequency;
        var take = block.StepInput * block.Frequency;

        for (var c = 0; c < block.Channels; c++)
        {
            Array.Copy(cache, (c * plane) + plane - take, destination, c * take, take);
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        for (var b = 0; b < BlockCount; b++)
        {
            _stepInputs[b][0].Dispose();
            _stepOutputs[b][0].Dispose();
            _sessions[b].Dispose();
        }

        _runOptions.Dispose();
    }
}
