using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

using Shouldly;

using Xunit;

namespace Personix.WakeWord.Tests;

/// <summary>
/// Pausing from outside: nothing is read, the source is told to let the microphone go, and
/// resuming starts from a clean window.
/// </summary>
public class WakeWordListenerPauseTests
{
    private static string? Models => Environment.GetEnvironmentVariable("WAKEWORD_MODELS");

    private static string? Samples => Environment.GetEnvironmentVariable("WAKEWORD_SAMPLES");

    [SkippableFact]
    public async Task While_paused_nothing_is_read_and_nothing_is_heard()
    {
        // Arrange — the source pauses the listener right before the word and never resumes
        var (audio, options) = Rumburaku();
        var heard = new List<WakeWordDetection>();
        var listener = Build(audio, options, new RecordingHandler(heard));
        audio.PauseListenerAfterFrames = 10;
        audio.Listener = listener;

        // Act — the file ends while paused, so cancel to finish
        using var cancellation = new CancellationTokenSource();
        var run = listener.RunAsync(cancellation.Token);
        await audio.PausedSignal.Task.WaitAsync(TimeSpan.FromSeconds(5));
        await Task.Delay(200);   // give a misbehaving loop time to read past the pause
        cancellation.Cancel();
        await run;

        // Assert
        audio.FramesRead.ShouldBe(10, "not one frame may be read while paused");
        audio.PauseCalls.ShouldBe(1);
        heard.ShouldBeEmpty();
    }

    [SkippableFact]
    public async Task After_resume_the_word_is_heard_again()
    {
        // Arrange — pause for a moment early in the file, then resume; the word comes later
        var (audio, options) = Rumburaku();
        var heard = new List<WakeWordDetection>();
        var listener = Build(audio, options, new RecordingHandler(heard));
        audio.PauseListenerAfterFrames = 5;
        audio.Listener = listener;

        // Act
        var run = listener.RunAsync(CancellationToken.None);
        await audio.PausedSignal.Task.WaitAsync(TimeSpan.FromSeconds(5));
        listener.State.ShouldBe(ListenerState.Paused);
        listener.Resume();
        await run;

        // Assert
        audio.ResumeCalls.ShouldBe(1);
        heard.ShouldHaveSingleItem().Word.ShouldBe("rumburaku");
        listener.State.ShouldBe(ListenerState.Stopped);
    }

    [Fact]
    public void Pausing_twice_and_resuming_when_not_paused_are_harmless()
    {
        // Arrange
        var audio = new FrameSource([]);
        var listener = Build(audio, WordOptions("word", "word.wwc"));

        // Act
        listener.Pause();
        listener.Pause();
        listener.Resume();
        listener.Resume();

        // Assert — the source heard each transition once
        audio.PauseCalls.ShouldBe(1);
        audio.ResumeCalls.ShouldBe(1);
        listener.State.ShouldBe(ListenerState.Stopped, "it never ran");
    }

    private static (FrameSource Source, WakeWordOptions Options) Rumburaku()
    {
        var models = Models;
        var samples = Samples;
        Skip.If(string.IsNullOrEmpty(models) || string.IsNullOrEmpty(samples), "set WAKEWORD_MODELS and WAKEWORD_SAMPLES");

        var recording = Path.Combine(samples, "rumburaku", "rumburaku_000.wav");
        Skip.IfNot(File.Exists(recording), "reference recording missing");

        return (new FrameSource(ReadPcm(recording)), WordOptions("rumburaku", Path.Combine(models, "rumburaku.wwc")));
    }

    private static WakeWordOptions WordOptions(string word, string path)
    {
        var options = new WakeWordOptions();
        options.Add(word, path, 0.9f);
        return options;
    }

    private static WakeWordListener Build(IAudioSource audio, WakeWordOptions options, params IWakeWordHandler[] handlers)
    {
        var services = new ServiceCollection();
        foreach (var handler in handlers)
        {
            services.AddScoped(_ => handler);
        }

        var provider = services.BuildServiceProvider();
        return new WakeWordListener(audio, Options.Create(options), provider.GetRequiredService<IServiceScopeFactory>(), NullLogger<WakeWordListener>.Instance);
    }

    /// <summary>Plays samples frame by frame and can pause the listener itself after a number of frames.</summary>
    private sealed class FrameSource(short[] samples) : IAudioSource
    {
        private int _position;

        public int FramesRead { get; private set; }

        public int PauseCalls { get; private set; }

        public int ResumeCalls { get; private set; }

        public int? PauseListenerAfterFrames { get; set; }

        public WakeWordListener? Listener { get; set; }

        public TaskCompletionSource PausedSignal { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public ValueTask<bool> ReadFrameAsync(Memory<short> frame, CancellationToken cancellationToken)
        {
            if (_position + frame.Length > samples.Length)
            {
                return ValueTask.FromResult(false);
            }

            samples.AsMemory(_position, frame.Length).CopyTo(frame);
            _position += frame.Length;
            FramesRead++;

            if (FramesRead == PauseListenerAfterFrames)
            {
                Listener!.Pause();
            }

            return ValueTask.FromResult(true);
        }

        public void Pause()
        {
            PauseCalls++;
            PausedSignal.TrySetResult();
        }

        public void Resume() => ResumeCalls++;
    }

    private sealed class RecordingHandler(List<WakeWordDetection> heard) : IWakeWordHandler
    {
        public Task HandleAsync(WakeWordDetection detection, CancellationToken cancellationToken)
        {
            heard.Add(detection);
            return Task.CompletedTask;
        }
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
                var result = new short[size / 2];
                Buffer.BlockCopy(bytes, at + 8, result, 0, size);
                return result;
            }

            at += 8 + size + (size % 2);
        }

        throw new InvalidDataException($"{path} has no data chunk");
    }
}
