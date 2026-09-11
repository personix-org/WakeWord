using Microsoft.ML.OnnxRuntime;

namespace Personix.WakeWord;

/// <summary>
/// A classifier from an ONNX file — the models that ship with openWakeWord.
///
/// Input and output buffers are allocated once and handed to the session as tensors over that
/// memory, so scoring a frame does not allocate. A window arrives laid out frame after frame,
/// which is exactly the shape the model wants, so it goes in with a single copy.
/// </summary>
internal sealed class OnnxWakeWordClassifier : IWakeWordClassifier
{
    private readonly InferenceSession _session;
    private readonly RunOptions _runOptions = new();

    private readonly float[] _input;
    private readonly float[] _output;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly OrtValue[] _inputs;
    private readonly OrtValue[] _outputs;

    public OnnxWakeWordClassifier(string path, SessionOptions options, int frames, int embeddingSize)
    {
        _session = new InferenceSession(path, options);
        _inputNames = [_session.InputMetadata.Keys.First()];
        _outputNames = [.. _session.OutputMetadata.Keys];

        InputSize = frames * embeddingSize;
        _input = new float[InputSize];
        _inputs = [OrtValue.CreateTensorValueFromMemory(_input, [1, frames, embeddingSize])];

        using (var probe = _session.Run(_runOptions, _inputNames, _inputs, _outputNames))
        {
                _output = probe[0].GetTensorDataAsSpan<float>().ToArray();
        }

        _outputs = [OrtValue.CreateTensorValueFromMemory(_output, [1, _output.Length])];
    }

    public int InputSize { get; }

    public float Score(ReadOnlySpan<float> window)
    {
        if (window.Length != InputSize)
        {
                throw new ArgumentException($"Expected {InputSize} values, got {window.Length}.", nameof(window));
        }

        window.CopyTo(_input);
        _session.Run(_runOptions, _inputNames, _inputs, _outputNames, _outputs);

        return _output[0];
    }

    public void Dispose()
    {
        foreach (var value in _inputs)
        {
                value.Dispose();
        }

        foreach (var value in _outputs)
        {
                value.Dispose();
        }

        _runOptions.Dispose();
        _session.Dispose();
    }
}
