using Shouldly;

using Xunit;

namespace Personix.WakeWord.Tests;

/// <summary>
/// Runs the whole chain and holds its numbers in place.
///
/// A trainer and a detector have to walk the same path, so a model is scored on the kind of
/// numbers it was trained on. The values below are a fingerprint of this chain taken on
/// 11 September 2026, with the log-mel front end as code (<see cref="Melspectrogram"/>) and the
/// embedding run incrementally (<see cref="SpeechEmbedding"/>). Against the original openWakeWord
/// chain the classifiers were trained through — ONNX front end, one pass per window — the
/// fingerprint moved by two parts in ten million and no score by more than 3e-7, measured over
/// 408 recordings with the same detections on every one of them.
///
/// The models are not in the repository — the pre-trained wake words that ship with openWakeWord
/// are licensed CC BY-NC-SA — so these tests run only when the environment points at a copy:
///
///   WAKEWORD_MODELS    directory with the .wwc classifiers the fingerprints were taken with
///   WAKEWORD_SAMPLES   directory with the reference recordings, one folder per word
///
/// The embedding itself comes with the package, copied next to the test binaries.
/// </summary>
public class WakeWordChainTests
{
    private static string? Models => Environment.GetEnvironmentVariable("WAKEWORD_MODELS");

    private static string? Samples => Environment.GetEnvironmentVariable("WAKEWORD_SAMPLES");

    /// <summary>Features must not move at all — the tolerance covers rounding, nothing more.</summary>
    private const double FeatureTolerance = 0.001;

    /// <summary>
    /// Scores may drift a little further: a dot product over SIMD sums in a different order than a
    /// value-at-a-time loop, which in float arithmetic shows up around the seventh digit.
    /// </summary>
    private const float ScoreTolerance = 1e-5f;

    [SkippableTheory]
    [InlineData("rumburaku", "rumburaku_000.wav", 21, 57726.7207, 1.0000000f)]
    [InlineData("rumburaku", "rumburaku_001.wav", 21, 54184.4672, 1.0000000f)]
    [InlineData("saturnine", "saturnine_001.wav", 9, 25716.3782, 1.0000000f)]
    [InlineData("saturnine", "saturnine_002.wav", 9, 23205.6347, 0.9999945f)]
    public void The_chain_gives_the_same_numbers_as_when_the_models_were_trained(
        string word, string recording, int expectedWindows, double expectedFeatureSum, float expectedScore)
    {
        // Arrange
        var models = RequireModels();
        var path = Path.Combine(RequireSamples(), word, recording);

        if (!File.Exists(path))
        {
            Skip.If(true, $"reference recording {recording} is not available");
        }

        var samples = ReadPcm(path);

        using var options = WakeWordFeatures.SequentialOptions();
        using var features = new WakeWordFeatures(WakeWordDetector.DefaultModelDirectory, options);
        using var classifier = WakeWordClassifiers.Open(
            Path.Combine(models, word + WakeWordClassifiers.DenseExtension),
            options, WakeWordFeatures.ClassifierFrames, WakeWordFeatures.EmbeddingSize);

        // Act
        var windows = 0;
        var featureSum = 0.0;
        var best = 0f;

        for (var offset = 0; offset + WakeWordFeatures.FrameLength <= samples.Length;
             offset += WakeWordFeatures.FrameLength)
        {
            var window = features.Process(samples.AsSpan(offset, WakeWordFeatures.FrameLength));
            if (window.IsEmpty)
            {
                continue;
            }

            windows++;
            foreach (var value in window)
            {
                featureSum += value;
            }

            best = Math.Max(best, classifier.Score(window));
        }

        // Assert
        windows.ShouldBe(expectedWindows, "the chain has to yield the same number of windows");
        featureSum.ShouldBe(expectedFeatureSum, FeatureTolerance,
            "neither the melspectrogram nor the embedding may shift");
        best.ShouldBe(expectedScore, ScoreTolerance);
    }

    /// <summary>
    /// The detector reports which word fired, so several wake words can share one chain and still
    /// be told apart.
    /// </summary>
    [SkippableFact]
    public void The_detector_reports_which_word_it_heard()
    {
        // Arrange
        var models = RequireModels();
        var path = Path.Combine(RequireSamples(), "rumburaku", "rumburaku_000.wav");

        if (!File.Exists(path))
        {
            Skip.If(true, "reference recording is not available");
        }

        using var detector = new WakeWordDetector(
            null,
            [Path.Combine(models, "saturnine.wwc"), Path.Combine(models, "rumburaku.wwc")],
            [0.9f, 0.9f]);

        var samples = ReadPcm(path);

        // Act
        var hits = new List<int>();
        for (var offset = 0; offset + WakeWordDetector.FrameLength <= samples.Length;
             offset += WakeWordDetector.FrameLength)
        {
            var hit = detector.Process(samples.AsSpan(offset, WakeWordDetector.FrameLength));
            if (hit >= 0)
            {
                hits.Add(hit);
            }
        }

        // Assert — index 1 is rumburaku, and saturnine must stay quiet on it
        hits.ShouldNotBeEmpty("the recording is of the wake word being said");
        hits.ShouldAllBe(hit => hit == 1);
    }

    [SkippableFact]
    public void Processing_a_frame_allocates_almost_nothing()
    {
        // Arrange
        var models = RequireModels();

        using var options = WakeWordFeatures.SequentialOptions();
        using var features = new WakeWordFeatures(WakeWordDetector.DefaultModelDirectory, options);

        var frame = new short[WakeWordFeatures.FrameLength];
        var random = new Random(42);
        for (var i = 0; i < frame.Length; i++)
        {
            frame[i] = (short)random.Next(-8000, 8000);
        }

        for (var i = 0; i < 60; i++)
        {
            features.Process(frame);    // fill the buffers and warm the JIT up
        }

        // Act
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 100; i++)
        {
            features.Process(frame);
        }

        var perFrame = (GC.GetAllocatedBytesForCurrentThread() - before) / 100;

        // Assert — a frame goes through this every 80 ms for as long as the process runs.
        // Not zero: ONNX Runtime keeps a few objects of its own around each of the six block runs.
        perFrame.ShouldBeLessThan(4096);
    }

    private static string RequireModels()
    {
        var models = Models;

        if (string.IsNullOrEmpty(models) || !Directory.Exists(models))
        {
            Skip.If(true, "set WAKEWORD_MODELS to a directory holding the shared openWakeWord models");
        }

        return models;
    }

    private static string RequireSamples()
    {
        var samples = Samples;

        if (string.IsNullOrEmpty(samples) || !Directory.Exists(samples))
        {
            Skip.If(true, "set WAKEWORD_SAMPLES to a directory holding the reference recordings");
        }

        return samples;
    }

    /// <summary>Minimal reader for the 16 kHz mono 16-bit WAV files the reference set is stored in.</summary>
    private static short[] ReadPcm(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var data = 12;

        while (data + 8 <= bytes.Length)
        {
            var id = System.Text.Encoding.ASCII.GetString(bytes, data, 4);
            var size = BitConverter.ToInt32(bytes, data + 4);

            if (id == "data")
            {
                var samples = new short[size / 2];
                Buffer.BlockCopy(bytes, data + 8, samples, 0, size);
                return samples;
            }

            data += 8 + size + (size % 2);
        }

        throw new InvalidDataException($"{path} has no data chunk");
    }
}
