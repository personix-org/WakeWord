namespace Personix.WakeWord;

/// <summary>
/// Detects wake words in a stream of audio, entirely on the device.
///
/// The chain follows the reference Python library (openwakeword):
/// <code>
/// 1280 samples → melspectrogram → speech embedding → classifier
/// </code>
///
/// The melspectrogram and the embedding are shared by every wake word and run once per frame
/// (see <see cref="WakeWordFeatures"/>). Adding another word therefore costs one small classifier
/// rather than a second chain.
///
/// Audio capture is left to the host. Feed <see cref="Process"/> consecutive frames of
/// <see cref="FrameLength"/> samples, 16 kHz mono PCM.
///
/// One instance belongs to one thread.
/// </summary>
/// <example>
/// <code>
/// using var detector = new WakeWordDetector(
///     modelDirectory: null,                       // the model the package installed
///     wakeWordModelPaths: ["my-word.wwc"],
///     thresholds: [0.9f]);
///
/// while (recorder.TryReadFrame(out var frame))
/// {
///     if (detector.Process(frame) >= 0)
///         Console.WriteLine("wake word heard");
/// }
/// </code>
/// </example>
public sealed class WakeWordDetector : IDisposable
{
    /// <summary>Samples per frame — 80 ms at 16 kHz.</summary>
    public const int FrameLength = WakeWordFeatures.FrameLength;

    /// <summary>
    /// Where the embedding model blocks land when the package is installed: a <c>models</c> folder
    /// next to the application's binaries.
    /// </summary>
    public static string DefaultModelDirectory => Path.Combine(AppContext.BaseDirectory, "models");

    private readonly WakeWordFeatures _features;
    private readonly IWakeWordClassifier[] _classifiers;
    private readonly float[] _thresholds;
    private readonly float[] _lastScores;

    /// <summary>
    /// Scores from 0 to 1 from the last frame, in the order the models were given. Useful for
    /// tuning thresholds and for logging near misses.
    /// </summary>
    public IReadOnlyList<float> LastScores => _lastScores;

    /// <param name="modelDirectory">
    /// Directory holding the embedding model blocks, <c>embedding_b0.onnx</c> to <c>embedding_b5.onnx</c>.
    /// Null means <see cref="DefaultModelDirectory"/>, which is where the package puts them when it is
    /// installed.
    /// </param>
    /// <param name="wakeWordModelPaths">
    /// Classifiers, in the order their index is reported back. The extension decides the format:
    /// <c>.onnx</c> are models from openWakeWord, <c>.wwc</c> are models from the TorchSharp trainer.
    /// </param>
    /// <param name="thresholds">
    /// Threshold per model. The library's own default is 0.5; a word that fires on similar-sounding
    /// speech wants a higher one.
    /// </param>
    public WakeWordDetector(
        string? modelDirectory,
        IReadOnlyList<string> wakeWordModelPaths,
        IReadOnlyList<float> thresholds)
    {
        ArgumentNullException.ThrowIfNull(wakeWordModelPaths);
        ArgumentNullException.ThrowIfNull(thresholds);

        if (wakeWordModelPaths.Count == 0)
        {
                throw new ArgumentException("At least one wake word model is needed.", nameof(wakeWordModelPaths));
        }

        if (thresholds.Count != wakeWordModelPaths.Count)
        {
            throw new ArgumentException(
                $"Got {wakeWordModelPaths.Count} models but {thresholds.Count} thresholds.", nameof(thresholds));
        }

        using var options = WakeWordFeatures.SequentialOptions();

        _features = new WakeWordFeatures(modelDirectory ?? DefaultModelDirectory, options);

        _classifiers = [.. wakeWordModelPaths.Select(path => WakeWordClassifiers.Open(
            path, options, WakeWordFeatures.ClassifierFrames, WakeWordFeatures.EmbeddingSize))];

        _thresholds = [.. thresholds];
        _lastScores = new float[_classifiers.Length];
    }

    /// <summary>
    /// Processes one frame. Returns the index of the wake word that fired, or -1 for none.
    ///
    /// For roughly the first 1.3 seconds after start it always returns -1, until the classifier
    /// window fills — scoring a half-empty buffer would fire on nothing.
    /// </summary>
    /// <param name="frame">Exactly <see cref="FrameLength"/> samples of 16 kHz mono PCM.</param>
    public int Process(ReadOnlySpan<short> frame)
    {
        var window = _features.Process(frame);
        if (window.IsEmpty)
        {
                return -1;
        }

        var hit = -1;

        for (var i = 0; i < _classifiers.Length; i++)
        {
            _lastScores[i] = _classifiers[i].Score(window);

            // The first model over its threshold wins, but every model gets its score recorded.
            if (hit < 0 && _lastScores[i] >= _thresholds[i])
            {
                    hit = i;
            }
        }

        return hit;
    }

    /// <summary>
    /// Forgets the audio heard so far. Call this between unrelated recordings, so the tail of one
    /// does not bleed into the next.
    /// </summary>
    public void Reset() => _features.Reset();

    /// <inheritdoc />
    public void Dispose()
    {
        _features.Dispose();

        foreach (var classifier in _classifiers)
        {
                classifier.Dispose();
        }
    }
}
