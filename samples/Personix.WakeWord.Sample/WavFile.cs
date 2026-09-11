namespace Personix.WakeWord.Sample;

/// <summary>
/// Reads 16-bit PCM WAV files, which is the format the detector expects once resampled to 16 kHz.
///
/// The library itself takes plain sample arrays and has no opinion on file formats or capture —
/// this reader exists so the sample can run on a recording instead of a microphone.
/// </summary>
public static class WavFile
{
    public const int SampleRate = 16000;

    /// <summary>Reads mono 16-bit PCM samples, skipping over any chunks before the data.</summary>
    public static short[] Read(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        if (new string(reader.ReadChars(4)) != "RIFF")
        {
            throw new InvalidDataException($"{path} is not a WAV file");
        }

        reader.ReadInt32();   // file size

        if (new string(reader.ReadChars(4)) != "WAVE")
        {
            throw new InvalidDataException($"{path} is not a WAV file");
        }

        short channels = 1;
        var sampleRate = SampleRate;
        short bitsPerSample = 16;

        while (stream.Position < stream.Length)
        {
            var chunkId = new string(reader.ReadChars(4));
            var chunkSize = reader.ReadInt32();

            if (chunkId == "fmt ")
            {
                reader.ReadInt16();                    // audio format
                channels = reader.ReadInt16();
                sampleRate = reader.ReadInt32();
                reader.ReadInt32();                    // byte rate
                reader.ReadInt16();                    // block align
                bitsPerSample = reader.ReadInt16();

                if (chunkSize > 16)
                {
                    stream.Seek(chunkSize - 16, SeekOrigin.Current);
                }
            }
            else if (chunkId == "data")
            {
                if (bitsPerSample != 16)
                {
                    throw new InvalidDataException($"{path} is {bitsPerSample}-bit, expected 16-bit PCM");
                }

                if (sampleRate != SampleRate)
                {
                    throw new InvalidDataException($"{path} is {sampleRate} Hz, expected {SampleRate} Hz");
                }

                var samples = new short[chunkSize / 2];
                for (var i = 0; i < samples.Length; i++)
                {
                    samples[i] = reader.ReadInt16();
                }

                return channels == 1 ? samples : ToMono(samples, channels);
            }
            else
            {
                stream.Seek(chunkSize, SeekOrigin.Current);
            }
        }

        throw new InvalidDataException($"{path} has no data chunk");
    }

    private static short[] ToMono(short[] interleaved, int channels)
    {
        var mono = new short[interleaved.Length / channels];

        for (var i = 0; i < mono.Length; i++)
        {
            var sum = 0;
            for (var c = 0; c < channels; c++)
            {
                sum += interleaved[(i * channels) + c];
            }

            mono[i] = (short)(sum / channels);
        }

        return mono;
    }
}
