using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace Personix.WakeWord;

/// <summary>
/// Pulls frames from the <see cref="IAudioSource"/>, runs them through a <see cref="WakeWordDetector"/>
/// and calls every <see cref="IWakeWordHandler"/> when a word fires. Runs as a hosted service for
/// as long as the source delivers audio.
///
/// A handler that throws is logged and does not stop the listener. A source that throws does —
/// recovering a lost microphone is the source's business, since only it knows how.
///
/// <see cref="Pause"/> and <see cref="Resume"/> may be called from any thread — a control
/// endpoint, a hotkey, another service. While paused the listener reads nothing and the source
/// is told to let the microphone go; on resume the detector starts from a clean window, so audio
/// from before the pause cannot combine with audio after it.
/// </summary>
public sealed class WakeWordListener(
    IAudioSource audio,
    IOptions<WakeWordOptions> options,
    IServiceScopeFactory scopes,
    ILogger<WakeWordListener> logger) : BackgroundService
{
    private static readonly TimeSpan FrameDuration =
        TimeSpan.FromSeconds((double)WakeWordDetector.FrameLength / Melspectrogram.SampleRate);

    private readonly Lock _gate = new();
    private TaskCompletionSource? _resumed;   // set while paused; completing it resumes the loop
    private volatile ListenerState _state = ListenerState.Stopped;

    /// <summary>What the listener is doing right now. Safe to read from any thread.</summary>
    public ListenerState State => _state;

    /// <summary>
    /// Stops reading until <see cref="Resume"/>. The source is told to let the microphone go.
    /// Calling it while already paused, or before the listener runs, does nothing.
    /// </summary>
    public void Pause()
    {
        lock (_gate)
        {
            if (_resumed is not null)
            {
                return;
            }

            _resumed = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        }

        audio.Pause();
        logger.LogInformation("Paused");
    }

    /// <summary>Starts reading again after <see cref="Pause"/>. Does nothing if not paused.</summary>
    public void Resume()
    {
        TaskCompletionSource resumed;

        lock (_gate)
        {
            if (_resumed is null)
            {
                return;
            }

            resumed = _resumed;
            _resumed = null;
        }

        audio.Resume();
        logger.LogInformation("Resumed");
        resumed.SetResult();
    }

    /// <summary>
    /// Listens until the source ends or the token is cancelled. This is what the hosted service
    /// runs; it is public so a host without <see cref="IHostedService"/> plumbing can drive it directly.
    /// </summary>
    public async Task RunAsync(CancellationToken cancellationToken)
    {
        var settings = options.Value;

        if (settings.Words.Count == 0)
        {
            throw new InvalidOperationException("No wake words registered — add at least one in AddWakeWord.");
        }

        using var detector = new WakeWordDetector(
            settings.ModelDirectory,
            [.. settings.Words.Select(w => w.Path)],
            [.. settings.Words.Select(w => w.Threshold)]);

        logger.LogInformation("Listening for {Words}", string.Join(", ", settings.Words.Select(w => $"{w.Word} ≥ {w.Threshold:F2}")));

        var frame = new short[WakeWordDetector.FrameLength];
        var frames = 0L;
        var quietUntil = TimeSpan.Zero;
        _state = ListenerState.Listening;

        try
        {
            while (true)
            {
                if (await WaitIfPausedAsync(cancellationToken).ConfigureAwait(false))
                {
                    // Whatever was in the detector's window is from before the pause.
                    detector.Reset();
                    quietUntil = TimeSpan.Zero;
                }

                if (!await audio.ReadFrameAsync(frame, cancellationToken).ConfigureAwait(false))
                {
                    break;
                }

                var at = frames * FrameDuration;
                frames++;

                var hit = detector.Process(frame);

                if (hit < 0 || at < quietUntil)
                {
                    continue;
                }

                quietUntil = at + settings.Cooldown;

                var detection = new WakeWordDetection(settings.Words[hit].Word, hit, detector.LastScores[hit], at);
                logger.LogInformation("Heard {Word} at {At} (score {Score:F3})", detection.Word, detection.At, detection.Score);

                await HandleAsync(detection, cancellationToken).ConfigureAwait(false);
            }

            logger.LogInformation("Audio source ended after {Frames} frames", frames);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            logger.LogInformation("Listener stopped after {Frames} frames", frames);
        }
        finally
        {
            _state = ListenerState.Stopped;
        }
    }

    /// <summary>Blocks while paused. Returns true if it did wait, so the caller can start afresh.</summary>
    private async Task<bool> WaitIfPausedAsync(CancellationToken cancellationToken)
    {
        Task? resumed;

        lock (_gate)
        {
            resumed = _resumed?.Task;
        }

        if (resumed is null)
        {
            return false;
        }

        _state = ListenerState.Paused;
        await resumed.WaitAsync(cancellationToken).ConfigureAwait(false);
        _state = ListenerState.Listening;
        return true;
    }

    /// <inheritdoc />
    protected override Task ExecuteAsync(CancellationToken stoppingToken) => RunAsync(stoppingToken);

    private async Task HandleAsync(WakeWordDetection detection, CancellationToken cancellationToken)
    {
        await using var scope = scopes.CreateAsyncScope();

        foreach (var handler in scope.ServiceProvider.GetServices<IWakeWordHandler>())
        {
            try
            {
                await handler.HandleAsync(detection, cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception e)
            {
                logger.LogError(e, "{Handler} failed on {Word}", handler, detection.Word);
            }
        }
    }
}
