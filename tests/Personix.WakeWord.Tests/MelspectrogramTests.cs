using Microsoft.ML.OnnxRuntime;

using Shouldly;

using Xunit;
using Xunit.Abstractions;

namespace Personix.WakeWord.Tests;

/// <summary>
/// The log-mel transform written here has to match the <c>melspectrogram.onnx</c> from openWakeWord,
/// because every classifier so far was trained on that model's numbers.
/// </summary>
public class MelspectrogramTests(ITestOutputHelper output)
{
    private static string? Models => Environment.GetEnvironmentVariable("WAKEWORD_MODELS");

    private static string? Samples => Environment.GetEnvironmentVariable("WAKEWORD_SAMPLES");

    [Theory]
    [InlineData(0, 0)]
    [InlineData(511, 0)]
    [InlineData(512, 1)]
    [InlineData(671, 1)]
    [InlineData(672, 2)]
    [InlineData(1280, 5)]
    [InlineData(1760, 8)]
    public void Frame_count_follows_the_reference_model(int samples, int frames)
    {
        Melspectrogram.FramesFor(samples).ShouldBe(frames);
    }

    [Fact]
    public void Silence_sits_on_the_floor()
    {
        // Arrange — digital silence has no energy, so every band is at the power floor
        var transform = new Melspectrogram();
        var silence = new float[1760];
        var result = new float[8 * Melspectrogram.MelBins];

        // Act
        var frames = transform.Compute(silence, result);

        // Assert — 10·log10(1e-10) = -100 dB everywhere, and the 80 dB floor changes nothing
        frames.ShouldBe(8);
        result.ShouldAllBe(v => Math.Abs(v - (-100f)) < 1e-4f);
    }

    [Fact]
    public void Output_that_is_too_small_is_refused()
    {
        var transform = new Melspectrogram();

        Action compute = () => transform.Compute(new float[1760], new float[10]);

        Should.Throw<ArgumentException>(compute);
    }

    /// <summary>
    /// Whole speech recordings, split the way the detector splits them — 1760 samples per call —
    /// compared value by value with the reference model.
    /// </summary>
    [SkippableTheory]
    [InlineData("rumburaku", "rumburaku_000.wav")]
    [InlineData("rumburaku", "rumburaku_001.wav")]
    [InlineData("saturnine", "saturnine_002.wav")]
    public void Matches_the_reference_model_on_speech(string word, string recording)
    {
        // Arrange
        var models = Models;
        var samples = Samples;
        Skip.If(string.IsNullOrEmpty(models) || string.IsNullOrEmpty(samples),
            "set WAKEWORD_MODELS and WAKEWORD_SAMPLES to run against the reference model");

        var referencePath = Path.Combine(models, "melspectrogram.onnx");
        var recordingPath = Path.Combine(samples, word, recording);
        Skip.IfNot(File.Exists(referencePath) && File.Exists(recordingPath), "reference model or recording missing");

        var audio = ReadPcm(recordingPath);
        var transform = new Melspectrogram();

        using var options = WakeWordFeatures.SequentialOptions();
        using var reference = new InferenceSession(referencePath, options);

        const int window = 1760;
        var ours = new float[8 * Melspectrogram.MelBins];
        var worst = 0f;
        var compared = 0;

        // Act — the same 1760-sample windows the feature chain feeds the model
        for (var offset = 0; offset + window <= audio.Length; offset += 1280)
        {
            var input = new float[window];
            for (var i = 0; i < window; i++)
            {
                input[i] = audio[offset + i];
            }

            var frames = transform.Compute(input, ours);

            using var inputValue = OrtValue.CreateTensorValueFromMemory(input, [1, window]);
            using var results = reference.Run(new RunOptions(), ["input"], [inputValue], ["output"]);
            var theirs = results[0].GetTensorDataAsSpan<float>();

            theirs.Length.ShouldBe(frames * Melspectrogram.MelBins);

            for (var i = 0; i < theirs.Length; i++)
            {
                worst = Math.Max(worst, Math.Abs(theirs[i] - ours[i]));
                compared++;
            }
        }

        output.WriteLine($"{recording}: {compared} values compared, largest difference {worst:E2} dB");

        // Assert — the reference runs in float32, this runs in double: the gap is rounding, nothing more
        compared.ShouldBeGreaterThan(0);
        worst.ShouldBeLessThan(1e-3f);
    }

    private static short[] ReadPcm(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var at = 12;

        while (at + 8 <= bytes.Length)
        {
            var id = System.Text.Encoding.ASCII.GetString(bytes, at, 4);
            var size = BitConverter.ToInt32(bytes, at + 4);

            if (id == "data")
            {
                var samples = new short[size / 2];
                Buffer.BlockCopy(bytes, at + 8, samples, 0, size);
                return samples;
            }

            at += 8 + size + (size % 2);
        }

        throw new InvalidDataException($"{path} has no data chunk");
    }
}
