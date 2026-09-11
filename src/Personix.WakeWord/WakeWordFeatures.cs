using Microsoft.ML.OnnxRuntime;

namespace Personix.WakeWord;

/// <summary>
/// The shared half of the openWakeWord chain: turns audio into windows of speech embeddings,
/// which is what a wake word classifier decides on.
///
/// <code>
/// 1280 samples → melspectrogram → speech embedding → window of 16 × 96
/// </code>
///
/// The melspectrogram is computed here (<see cref="Melspectrogram"/>); the embedding is a
/// pre-trained model that ships with the package and is never trained further, run incrementally
/// (<see cref="SpeechEmbedding"/>). Training and production therefore run the exact same path —
/// a classifier trained on numbers this stage did not produce would be scored on numbers it has
/// never seen.
///
/// Buffers and model inputs are allocated once and overwritten from then on, because in production
/// a frame goes through this every 80 milliseconds for as long as the process runs. One instance
/// therefore belongs to one thread.
/// </summary>
public sealed class WakeWordFeatures : IDisposable
{
    /// <summary>Samples per frame — 80 ms at 16 kHz.</summary>
    public const int FrameLength = 1280;

    /// <summary>How many embedding frames the classifier reads — a window of 1.28 s.</summary>
    public const int ClassifierFrames = 16;

    /// <summary>Length of a single speech embedding.</summary>
    public const int EmbeddingSize = 96;

    private const int MelBins = Melspectrogram.MelBins;
    private const int MelWindow = SpeechEmbedding.WindowFrames;

    // The melspectrogram is computed over a window longer than the frame itself: it needs the
    // overlap with preceding audio, otherwise artefacts appear at every frame boundary.
    private const int MelOverlap = 160 * 3;
    private const int MelInputLength = FrameLength + MelOverlap;

    /// <summary>Mel frames held by the ring buffer — the embedding window plus one run of headroom.</summary>
    private const int MelCapacity = 256;

    /// <summary>Embeddings held by the ring buffer — the classifier window plus headroom.</summary>
    private const int FeatureCapacity = 64;

    private readonly Melspectrogram _melspectrogram = new();
    private readonly SpeechEmbedding _embedding;

    private readonly short[] _raw = new short[MelInputLength];
    private int _rawCount;

    private readonly float[] _melInput = new float[MelInputLength];
    private readonly float[] _melOutput = new float[SpeechEmbedding.HopFrames * MelBins];
    private readonly float[] _mel = new float[MelCapacity * MelBins];
    private int _melHead;
    private int _melCount;

    private readonly float[] _embeddingStep = new float[SpeechEmbedding.StepFrames * MelBins];
    private readonly float[] _features = new float[FeatureCapacity * EmbeddingSize];
    private int _featureHead;
    private int _featureCount;

    private readonly float[] _window = new float[ClassifierFrames * EmbeddingSize];

    /// <summary>Values in one classifier window — 16 frames of 96.</summary>
    public static int WindowSize => ClassifierFrames * EmbeddingSize;

    /// <param name="modelDirectory">Directory holding the embedding model blocks, <c>embedding_b0.onnx</c> to <c>embedding_b5.onnx</c>.</param>
    /// <param name="options">Session settings — see <see cref="SequentialOptions"/>.</param>
    public WakeWordFeatures(string modelDirectory, SessionOptions options)
    {
        _embedding = new SpeechEmbedding(modelDirectory, options);

        Reset();
    }

    /// <summary>
    /// Processes one frame and returns the classifier window, or an empty span while the buffers
    /// are still filling — for roughly the first 1.3 seconds it returns nothing.
    ///
    /// The span points into a buffer that the next frame overwrites. Callers that need to keep a
    /// window should use <see cref="ProcessToArray"/>.
    /// </summary>
    /// <param name="frame">Exactly <see cref="FrameLength"/> samples of 16 kHz mono PCM.</param>
    public ReadOnlySpan<float> Process(ReadOnlySpan<short> frame)
    {
        if (frame.Length != FrameLength)
        {
                throw new ArgumentException($"Expected {FrameLength} samples, got {frame.Length}.", nameof(frame));
        }

        AppendRaw(frame);
        if (_rawCount < MelInputLength)
        {
                return default;
        }

        for (var i = 0; i < MelInputLength; i++)
        {
                _melInput[i] = _raw[i];
        }

        AppendMelspectrogram();
        if (_melCount < MelWindow)
        {
                return default;
        }

        AppendEmbedding();
        if (_featureCount < ClassifierFrames)
        {
                return default;
        }

        var start = (_featureHead - ClassifierFrames + FeatureCapacity) % FeatureCapacity;

        for (var t = 0; t < ClassifierFrames; t++)
        {
                FeatureSlot((start + t) % FeatureCapacity).CopyTo(_window.AsSpan(t * EmbeddingSize));
        }

        return _window;
    }

    /// <summary>
    /// Same as <see cref="Process"/>, but returns the window as an array of its own. For training,
    /// where windows are collected into a set and have to outlive the frames that follow.
    /// </summary>
    public float[]? ProcessToArray(ReadOnlySpan<short> frame)
    {
        var window = Process(frame);
        return window.IsEmpty ? null : window.ToArray();
    }

    /// <summary>
    /// Forgets everything heard so far. Training calls this between recordings, so the tail of one
    /// does not bleed into the next.
    /// </summary>
    public void Reset()
    {
        _rawCount = 0;
        _melHead = 0;
        _melCount = 0;
        _featureHead = 0;
        _featureCount = 0;

        // The reference implementation starts with the mel buffer filled with ones.
        for (var i = 0; i < MelWindow; i++)
        {
            MelSlot(_melHead).Fill(1f);
            Advance(ref _melHead, ref _melCount, MelCapacity);
        }

        // The embedding starts from the same window, so its first step continues from it.
        _embedding.Reset(_mel.AsSpan(0, MelWindow * MelBins));
    }

    /// <summary>Sliding window over raw audio — keeps the last <see cref="MelInputLength"/> samples.</summary>
    private void AppendRaw(ReadOnlySpan<short> frame)
    {
        if (_rawCount + frame.Length > MelInputLength)
        {
            var keep = MelInputLength - frame.Length;
            Array.Copy(_raw, _rawCount - keep, _raw, 0, keep);
            _rawCount = keep;
        }

        frame.CopyTo(_raw.AsSpan(_rawCount));
        _rawCount += frame.Length;
    }

    private void AppendMelspectrogram()
    {
        var frames = _melspectrogram.Compute(_melInput, _melOutput);

        for (var f = 0; f < frames; f++)
        {
            var row = MelSlot(_melHead);

            // transform from the reference implementation
            for (var b = 0; b < MelBins; b++)
            {
                row[b] = (_melOutput[(f * MelBins) + b] / 10f) + 2f;
            }

            Advance(ref _melHead, ref _melCount, MelCapacity);
        }
    }

    /// <summary>Moves the embedding on by the frames the melspectrogram just added — the last <see cref="SpeechEmbedding.StepFrames"/> of the buffer.</summary>
    private void AppendEmbedding()
    {
        var start = (_melHead - SpeechEmbedding.StepFrames + MelCapacity) % MelCapacity;

        for (var t = 0; t < SpeechEmbedding.StepFrames; t++)
        {
            MelSlot((start + t) % MelCapacity).CopyTo(_embeddingStep.AsSpan(t * MelBins));
        }

        _embedding.Advance(_embeddingStep).CopyTo(FeatureSlot(_featureHead));
        Advance(ref _featureHead, ref _featureCount, FeatureCapacity);
    }

    private Span<float> MelSlot(int index) => _mel.AsSpan(index * MelBins, MelBins);

    private Span<float> FeatureSlot(int index) => _features.AsSpan(index * EmbeddingSize, EmbeddingSize);

    private static void Advance(ref int head, ref int count, int capacity)
    {
        head = (head + 1) % capacity;

        if (count < capacity)
        {
                count++;
        }
    }

    /// <summary>
    /// Session settings for wake word detection: one thread, no spinning. The models are small
    /// enough that a thread pool costs more in synchronisation than it saves, and the process
    /// spends most of its life waiting for the next 80 ms of audio.
    /// </summary>
    public static SessionOptions SequentialOptions()
    {
        var options = new SessionOptions
        {
            IntraOpNumThreads = 1,
            InterOpNumThreads = 1,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
        };

        options.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
        options.AddSessionConfigEntry("session.inter_op.allow_spinning", "0");

        return options;
    }

    /// <inheritdoc />
    public void Dispose() => _embedding.Dispose();
}
