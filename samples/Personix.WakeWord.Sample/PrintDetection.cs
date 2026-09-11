namespace Personix.WakeWord.Sample;

/// <summary>What this sample does when a word is heard: says so.</summary>
public sealed class PrintDetection : IWakeWordHandler
{
    public Task HandleAsync(WakeWordDetection detection, CancellationToken cancellationToken)
    {
        Console.WriteLine($"  {detection.At.TotalSeconds,6:F2} s  {detection.Word}  (score {detection.Score:F4})");
        return Task.CompletedTask;
    }
}
