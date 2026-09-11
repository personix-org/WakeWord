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
/// </summary>
public sealed class WakeWordListener(
    IAudioSource audio,
    IOptions<WakeWordOptions> options,
    IServiceScopeFactory scopes,
    ILogger<WakeWordListener> logger) : BackgroundService
{
    private static readonly TimeSpan FrameDuration =
        TimeSpan.FromSeconds((double)WakeWordDetector.FrameLength / Melspectrogram.SampleRate);

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

        try
        {
            while (await audio.ReadFrameAsync(frame, cancellationToken).ConfigureAwait(false))
            {
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
