namespace Personix.WakeWord;

/// <summary>
/// Where the audio comes from. The host implements this over whatever it captures with — a
/// microphone library, a file, a network stream — and the listener pulls frames from it.
/// </summary>
public interface IAudioSource
{
    /// <summary>
    /// Fills <paramref name="frame"/> with the next <see cref="WakeWordDetector.FrameLength"/>
    /// samples of 16 kHz mono PCM. Returns false once the source has nothing more to give — a
    /// file that has been read to the end — which stops the listener.
    /// </summary>
    ValueTask<bool> ReadFrameAsync(Memory<short> frame, CancellationToken cancellationToken);
}
