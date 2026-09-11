namespace Personix.WakeWord.Sample;

/// <summary>
/// An audio source that plays a WAV file frame by frame, then ends. A microphone source would
/// look the same from the outside, only it would never return false.
/// </summary>
public sealed class WavFileAudioSource(string path) : IAudioSource
{
    private readonly short[] _samples = WavFile.Read(path);
    private int _position;

    public ValueTask<bool> ReadFrameAsync(Memory<short> frame, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (_position + frame.Length > _samples.Length)
        {
            return ValueTask.FromResult(false);
        }

        _samples.AsMemory(_position, frame.Length).CopyTo(frame);
        _position += frame.Length;
        return ValueTask.FromResult(true);
    }
}
