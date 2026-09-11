namespace Personix.WakeWord;

/// <summary>
/// What happens when a wake word is heard. Register as many as needed with
/// <see cref="WakeWordBuilder.AddHandler{THandler}"/>; each detection resolves them in a fresh
/// scope and calls them in registration order.
///
/// The listener waits for the handler, so audio piles up in the source meanwhile. A handler that
/// has long work to do should hand it off and return.
/// </summary>
public interface IWakeWordHandler
{
    Task HandleAsync(WakeWordDetection detection, CancellationToken cancellationToken);
}
