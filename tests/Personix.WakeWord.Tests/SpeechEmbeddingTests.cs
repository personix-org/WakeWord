using Microsoft.ML.OnnxRuntime;

using Shouldly;

using Xunit;
using Xunit.Abstractions;

namespace Personix.WakeWord.Tests;

/// <summary>
/// The embedding run incrementally has to be the same function as one pass through the whole
/// model. The whole model is not in the repository, so the comparison runs only where the
/// environment points at a copy of openWakeWord's <c>embedding_model.onnx</c>.
/// </summary>
public class SpeechEmbeddingTests(ITestOutputHelper output)
{
    private static string? Models => Environment.GetEnvironmentVariable("WAKEWORD_MODELS");

    /// <summary>The blocks the package ships, copied next to the test binaries by the project reference.</summary>
    private static string Blocks => WakeWordDetector.DefaultModelDirectory;

    private const int Steps = 40;

    [SkippableFact]
    public void Advancing_gives_the_same_embedding_as_a_full_pass()
    {
        // Arrange
        var models = Models;
        Skip.If(string.IsNullOrEmpty(models) || !File.Exists(Path.Combine(models, "embedding_model.onnx")),
            "set WAKEWORD_MODELS to a directory holding openWakeWord's embedding_model.onnx");

        using var options = WakeWordFeatures.SequentialOptions();
        using var whole = new InferenceSession(Path.Combine(models, "embedding_model.onnx"), options);
        using var streaming = new SpeechEmbedding(Blocks, options);
        using var runOptions = new RunOptions();

        var random = new Random(12345);
        var frames = SpeechEmbedding.WindowFrames + (SpeechEmbedding.HopFrames * Steps);
        var stream = new float[frames * SpeechEmbedding.MelBins];
        for (var i = 0; i < stream.Length; i++)
        {
            stream[i] = (float)(random.NextDouble() * 4);
        }

        streaming.Reset(stream.AsSpan(0, SpeechEmbedding.WindowFrames * SpeechEmbedding.MelBins));

        var worst = 0f;
        var largest = 0f;

        // Act — every step: the whole model on the new window against the incremental one
        for (var step = 1; step <= Steps; step++)
        {
            var first = step * SpeechEmbedding.HopFrames;
            var window = stream.AsSpan(first * SpeechEmbedding.MelBins, SpeechEmbedding.WindowFrames * SpeechEmbedding.MelBins).ToArray();

            using var input = OrtValue.CreateTensorValueFromMemory(window, [1, SpeechEmbedding.WindowFrames, SpeechEmbedding.MelBins, 1]);
            using var results = whole.Run(runOptions, ["input_1"], [input], [.. whole.OutputMetadata.Keys]);
            var expected = results[0].GetTensorDataAsSpan<float>();

            var tail = (SpeechEmbedding.WindowFrames - SpeechEmbedding.StepFrames) * SpeechEmbedding.MelBins;
            var actual = streaming.Advance(window.AsSpan(tail));

            for (var i = 0; i < SpeechEmbedding.Size; i++)
            {
                worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]));
                largest = Math.Max(largest, Math.Abs(expected[i]));
            }
        }

        output.WriteLine($"{Steps} steps, values up to ±{largest:F0}, largest difference {worst:E2}");

        // Assert — convolutions over different shapes sum in a different order; that is all the gap may be
        worst.ShouldBeLessThan(1e-3f);
    }

    [Fact]
    public void Reset_makes_the_stream_start_over()
    {
        // Arrange
        using var options = WakeWordFeatures.SequentialOptions();
        using var embedding = new SpeechEmbedding(Blocks, options);

        var random = new Random(7);
        var window = new float[SpeechEmbedding.WindowFrames * SpeechEmbedding.MelBins];
        var step = new float[SpeechEmbedding.StepFrames * SpeechEmbedding.MelBins];
        for (var i = 0; i < window.Length; i++)
        {
            window[i] = (float)random.NextDouble();
        }

        for (var i = 0; i < step.Length; i++)
        {
            step[i] = (float)random.NextDouble();
        }

        // Act
        embedding.Reset(window);
        var first = embedding.Advance(step).ToArray();
        embedding.Advance(step);
        embedding.Advance(step);

        embedding.Reset(window);
        var again = embedding.Advance(step).ToArray();

        // Assert
        again.ShouldBe(first);
    }

    [Fact]
    public void Advancing_does_not_allocate()
    {
        // Arrange
        using var options = WakeWordFeatures.SequentialOptions();
        using var embedding = new SpeechEmbedding(Blocks, options);

        var window = new float[SpeechEmbedding.WindowFrames * SpeechEmbedding.MelBins];
        Array.Fill(window, 1f);
        var step = new float[SpeechEmbedding.StepFrames * SpeechEmbedding.MelBins];
        Array.Fill(step, 2f);

        embedding.Reset(window);
        for (var i = 0; i < 30; i++)
        {
            embedding.Advance(step);
        }

        // Act
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 100; i++)
        {
            embedding.Advance(step);
        }

        var perStep = (GC.GetAllocatedBytesForCurrentThread() - before) / 100;

        // Assert — six model runs a step, and ONNX Runtime keeps a few objects around each
        perStep.ShouldBeLessThan(4096);
    }

    [Theory]
    [InlineData(10)]
    [InlineData(76 * 32 + 1)]
    public void A_window_of_the_wrong_size_is_refused(int length)
    {
        using var options = WakeWordFeatures.SequentialOptions();
        using var embedding = new SpeechEmbedding(Blocks, options);

        Action reset = () => embedding.Reset(new float[length]);

        Should.Throw<ArgumentException>(reset);
    }
}
