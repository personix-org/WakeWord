namespace Personix.WakeWord;

/// <summary>
/// Wraps a handler registered for some words only, and lets through just those. The listener
/// sees an ordinary handler; the filtering is invisible to it.
/// </summary>
internal sealed class WordFilteredHandler(IWakeWordHandler inner, IReadOnlySet<string> words) : IWakeWordHandler
{
    public Task HandleAsync(WakeWordDetection detection, CancellationToken cancellationToken) =>
        words.Contains(detection.Word)
            ? inner.HandleAsync(detection, cancellationToken)
            : Task.CompletedTask;

    /// <summary>The wrapped handler's name, so logs name what the host registered.</summary>
    public override string ToString() => inner.GetType().Name;
}
