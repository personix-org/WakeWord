namespace Personix.WakeWord;

/// <summary>Settings for the listener — which words, from where, and how it behaves once one fires.</summary>
public sealed class WakeWordOptions
{
    /// <summary>
    /// Directory holding the embedding model blocks. Null means the ones the package installed,
    /// see <see cref="WakeWordDetector.DefaultModelDirectory"/>.
    /// </summary>
    public string? ModelDirectory { get; set; }

    /// <summary>
    /// How long after a detection further hits are ignored. A word said once scores above the
    /// threshold on several consecutive frames; this turns them into one detection.
    /// </summary>
    public TimeSpan Cooldown { get; set; } = TimeSpan.FromMilliseconds(800);

    /// <summary>The words to listen for, in the order <see cref="WakeWordDetection.Index"/> counts them.</summary>
    public IList<WakeWordModel> Words { get; } = [];

    /// <summary>Adds a word to listen for.</summary>
    /// <param name="word">A name for it — what <see cref="WakeWordDetection.Word"/> reports.</param>
    /// <param name="path">The classifier file, <c>.wwc</c> or <c>.onnx</c>.</param>
    /// <param name="threshold">Score from which it counts as heard. The library's own default is 0.5.</param>
    public WakeWordOptions Add(string word, string path, float threshold = 0.5f)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(word);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        if (threshold is < 0f or > 1f || float.IsNaN(threshold))
        {
            throw new ArgumentOutOfRangeException(nameof(threshold), threshold, "A threshold is a score between 0 and 1.");
        }

        Words.Add(new WakeWordModel(word, path, threshold));
        return this;
    }
}
