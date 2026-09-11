using Microsoft.ML.OnnxRuntime;

namespace Personix.WakeWord;

/// <summary>Opens a wake word classifier, picking the reader by file extension.</summary>
public static class WakeWordClassifiers
{
    /// <summary>Extension of the trainer's own format (see <see cref="DenseClassifier"/>).</summary>
    public const string DenseExtension = ".wwc";

    /// <param name="path">
    /// Model file. A <c>.wwc</c> is read as a <see cref="DenseClassifier"/>, anything else as ONNX.
    /// </param>
    /// <param name="options">Session settings — see <see cref="WakeWordFeatures.SequentialOptions"/>.</param>
    /// <param name="frames">Embedding frames per window.</param>
    /// <param name="embeddingSize">Length of one embedding.</param>
    public static IWakeWordClassifier Open(string path, SessionOptions options, int frames, int embeddingSize)
    {
        if (!File.Exists(path))
        {
                throw new FileNotFoundException($"wake word model not found: {path}", path);
        }

        return Path.GetExtension(path).Equals(DenseExtension, StringComparison.OrdinalIgnoreCase)
            ? new DenseWakeWordClassifier(DenseClassifier.Load(path), frames * embeddingSize)
            : new OnnxWakeWordClassifier(path, options, frames, embeddingSize);
    }
}
