namespace Personix.WakeWord;

/// <summary>
/// The last link of the chain: given a window of speech embeddings, says how much it looks like
/// the wake word.
///
/// There are two shapes of it, because the model format depends on who trained it. Models that
/// ship with openWakeWord are ONNX. Models trained with the TorchSharp trainer are stored as a
/// <see cref="DenseClassifier"/>, since TorchSharp cannot export to ONNX.
/// </summary>
public interface IWakeWordClassifier : IDisposable
{
    /// <summary>Values expected on the input — a window of 16 frames of 96 embeddings.</summary>
    int InputSize { get; }

    /// <summary>Probability from 0 to 1 that the window holds the wake word.</summary>
    float Score(ReadOnlySpan<float> window);
}
