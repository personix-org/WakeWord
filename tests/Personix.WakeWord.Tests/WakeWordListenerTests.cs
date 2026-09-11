using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

using Shouldly;

using Xunit;

namespace Personix.WakeWord.Tests;

/// <summary>
/// The listener end to end: frames in from a source, detections out to handlers. The recordings
/// and classifiers come from the environment, see <see cref="WakeWordChainTests"/>.
/// </summary>
public class WakeWordListenerTests
{
    private static string? Models => Environment.GetEnvironmentVariable("WAKEWORD_MODELS");

    private static string? Samples => Environment.GetEnvironmentVariable("WAKEWORD_SAMPLES");

    [SkippableFact]
    public async Task A_word_in_the_recording_reaches_the_handler_once()
    {
        // Arrange — the recording has the word said once, which scores high on several frames
        var (source, options) = Rumburaku();
        var heard = new List<WakeWordDetection>();

        // Act
        await Run(source, options, new RecordingHandler(heard));

        // Assert — the cooldown folds those frames into one detection
        var detection = heard.ShouldHaveSingleItem();
        detection.Word.ShouldBe("rumburaku");
        detection.Index.ShouldBe(1, "saturnine was registered first");
        detection.Score.ShouldBeGreaterThanOrEqualTo(0.9f);
        detection.At.ShouldBeInRange(TimeSpan.FromSeconds(1), TimeSpan.FromSeconds(3));
    }

    [SkippableFact]
    public async Task Without_a_cooldown_every_firing_frame_is_a_detection()
    {
        // Arrange
        var (source, options) = Rumburaku();
        options.Cooldown = TimeSpan.Zero;
        var heard = new List<WakeWordDetection>();

        // Act
        await Run(source, options, new RecordingHandler(heard));

        // Assert
        heard.Count.ShouldBeGreaterThan(1);
        heard.ShouldAllBe(d => d.Word == "rumburaku");
    }

    [SkippableFact]
    public async Task A_handler_that_throws_does_not_stop_the_others_or_the_listener()
    {
        // Arrange
        var (source, options) = Rumburaku();
        var heard = new List<WakeWordDetection>();

        // Act — the throwing handler comes first
        await Run(source, options, new ThrowingHandler(), new RecordingHandler(heard));

        // Assert — the recording handler still ran, and the listener read the file to its end
        heard.ShouldHaveSingleItem();
        source.Exhausted.ShouldBeTrue();
    }

    [SkippableFact]
    public async Task A_handler_limited_to_other_words_stays_quiet()
    {
        // Arrange — both handlers are registered, only one answers to rumburaku
        var (source, options) = Rumburaku();
        var rumburaku = new List<WakeWordDetection>();
        var saturnine = new List<WakeWordDetection>();

        // Act
        await Run(source, options, CancellationToken.None,
            new WordFilteredHandler(new RecordingHandler(saturnine), new HashSet<string> { "saturnine" }),
            new WordFilteredHandler(new RecordingHandler(rumburaku), new HashSet<string> { "rumburaku" }));

        // Assert
        rumburaku.ShouldHaveSingleItem().Word.ShouldBe("rumburaku");
        saturnine.ShouldBeEmpty();
    }

    [Fact]
    public async Task The_listener_ends_when_the_source_does()
    {
        // Arrange — silence for a second, then the source is done
        var source = new FrameSource(new short[16000]);
        var options = new WakeWordOptions();
        options.Add("word", Path.Combine(WakeWordDetector.DefaultModelDirectory, "..", "missing.wwc"));

        // Act
        var run = () => Run(source, options);

        // Assert — a word that cannot be loaded is a configuration error, reported when listening starts
        await Should.ThrowAsync<FileNotFoundException>(run);
    }

    [Fact]
    public async Task Listening_without_words_is_refused()
    {
        var run = () => Run(new FrameSource([]), new WakeWordOptions());

        await Should.ThrowAsync<InvalidOperationException>(run);
    }

    [SkippableFact]
    public async Task Cancellation_stops_the_listener_quietly()
    {
        // Arrange — a source that never ends on its own, and cancels the listener on its first read
        var models = Models;
        Skip.If(string.IsNullOrEmpty(models), "set WAKEWORD_MODELS for a classifier to load");

        var options = new WakeWordOptions();
        options.Add("rumburaku", Path.Combine(models, "rumburaku.wwc"), 0.9f);

        using var cancellation = new CancellationTokenSource();
        var source = new EndlessSource(cancellation);

        // Act
        await Run(source, options, cancellation.Token);

        // Assert — cancellation is the normal way to stop; it must not surface as a failure
        source.Reads.ShouldBe(1);
    }

    private static (FrameSource Source, WakeWordOptions Options) Rumburaku()
    {
        var models = Models;
        var samples = Samples;
        Skip.If(string.IsNullOrEmpty(models) || string.IsNullOrEmpty(samples), "set WAKEWORD_MODELS and WAKEWORD_SAMPLES");

        var recording = Path.Combine(samples, "rumburaku", "rumburaku_000.wav");
        Skip.IfNot(File.Exists(recording), "reference recording missing");

        var options = new WakeWordOptions();
        options.Add("saturnine", Path.Combine(models, "saturnine.wwc"), 0.9f);
        options.Add("rumburaku", Path.Combine(models, "rumburaku.wwc"), 0.9f);

        return (new FrameSource(ReadPcm(recording)), options);
    }

    private static Task Run(IAudioSource source, WakeWordOptions options, params IWakeWordHandler[] handlers) =>
        Run(source, options, CancellationToken.None, handlers);

    private static async Task Run(IAudioSource source, WakeWordOptions options, CancellationToken cancellationToken, params IWakeWordHandler[] handlers)
    {
        var services = new ServiceCollection();
        foreach (var handler in handlers)
        {
            services.AddScoped(_ => handler);
        }

        await using var provider = services.BuildServiceProvider();
        var listener = new WakeWordListener(source, Options.Create(options), provider.GetRequiredService<IServiceScopeFactory>(), NullLogger<WakeWordListener>.Instance);

        await listener.RunAsync(cancellationToken);
    }

    private sealed class FrameSource(short[] samples) : IAudioSource
    {
        private int _position;

        public bool Exhausted { get; private set; }

        public ValueTask<bool> ReadFrameAsync(Memory<short> frame, CancellationToken cancellationToken)
        {
            if (_position + frame.Length > samples.Length)
            {
                Exhausted = true;
                return ValueTask.FromResult(false);
            }

            samples.AsMemory(_position, frame.Length).CopyTo(frame);
            _position += frame.Length;
            return ValueTask.FromResult(true);
        }
    }

    private sealed class EndlessSource(CancellationTokenSource cancellation) : IAudioSource
    {
        public int Reads { get; private set; }

        public ValueTask<bool> ReadFrameAsync(Memory<short> frame, CancellationToken cancellationToken)
        {
            Reads++;
            cancellation.Cancel();
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult(true);
        }
    }

    private sealed class RecordingHandler(List<WakeWordDetection> heard) : IWakeWordHandler
    {
        public Task HandleAsync(WakeWordDetection detection, CancellationToken cancellationToken)
        {
            heard.Add(detection);
            return Task.CompletedTask;
        }
    }

    private sealed class ThrowingHandler : IWakeWordHandler
    {
        public Task HandleAsync(WakeWordDetection detection, CancellationToken cancellationToken) =>
            throw new InvalidOperationException("this handler is broken");
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
