using System.Globalization;

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

using Personix.WakeWord;
using Personix.WakeWord.Sample;

// Listens for wake words in a recording and prints every one it hears.
//
//   dotnet run -- <recording.wav> <model>[:threshold] [<model>[:threshold] …]
//
// The recording is 16 kHz mono WAV. Each model is one classifier, .wwc or .onnx; the shared
// embedding model comes with the package and is found next to the binaries.
//
// This is the same wiring a real application uses — AddWakeWord, an audio source, a handler —
// only the audio source reads a file instead of a microphone, so the listener ends with it.

if (args.Length < 2)
{
    Console.Error.WriteLine(
        """
        usage: wakeword-sample <recording.wav> <model>[:threshold] …

          recording.wav     16 kHz mono, 16-bit PCM
          model             a .onnx or .wwc classifier, with an optional threshold (default 0.5)

        example:
          wakeword-sample ./hello.wav ./models/my-word.wwc:0.9
        """);
    return 1;
}

var recording = args[0];

var services = new ServiceCollection();
services.AddLogging(logging => logging.AddSimpleConsole(console => console.SingleLine = true).SetMinimumLevel(LogLevel.Information));

services.AddWakeWord(words =>
    {
        foreach (var argument in args.Skip(1))
        {
            var (path, threshold) = Parse(argument);
            words.Add(Path.GetFileNameWithoutExtension(path), path, threshold);
        }
    })
    .AddAudioSource(_ => new WavFileAudioSource(recording))
    .AddHandler<PrintDetection>();

await using var provider = services.BuildServiceProvider();

try
{
    await provider.GetRequiredService<WakeWordListener>().RunAsync(CancellationToken.None);
}
catch (Exception e) when (e is IOException or InvalidDataException)
{
    Console.Error.WriteLine($"cannot read {recording}: {e.Message}");
    return 1;
}

return 0;

// "path:0.9" — a Windows drive letter also carries a colon, so only a parsable tail counts.
static (string Path, float Threshold) Parse(string argument)
{
    var separator = argument.LastIndexOf(':');

    return separator > 1
        && float.TryParse(argument[(separator + 1)..], NumberStyles.Float, CultureInfo.InvariantCulture, out var threshold)
        ? (argument[..separator], threshold)
        : (argument, 0.5f);
}
