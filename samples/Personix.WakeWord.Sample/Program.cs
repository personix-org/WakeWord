using System.Globalization;

using Personix.WakeWord;
using Personix.WakeWord.Sample;

// Runs the detector over a recording and reports every wake word it hears.
//
//   dotnet run -- <model-directory> <recording.wav> <model>[:threshold] [<model>[:threshold] …]
//
// The model directory is the one holding melspectrogram.onnx and embedding_model.onnx. Those two
// are shared by every wake word; each further model is one small classifier on top of them.
//
// Both models and audio come from outside the library: the package ships code, not weights, and
// capture is the host's business. Here the audio is a 16 kHz mono WAV.

if (args.Length < 3)
{
    Console.Error.WriteLine(
        """
        usage: wakeword-sample <model-directory> <recording.wav> <model>[:threshold] …

          model-directory   holds melspectrogram.onnx and embedding_model.onnx
          recording.wav     16 kHz mono, 16-bit PCM
          model             a .onnx or .wwc classifier, with an optional threshold (default 0.5)

        example:
          wakeword-sample ./models ./hello.wav ./models/hey_jarvis_v0.1.onnx:0.5
        """);
    return 1;
}

var modelDirectory = args[0];
var recording = args[1];

var paths = new List<string>();
var thresholds = new List<float>();

foreach (var argument in args.Skip(2))
{
    var separator = argument.LastIndexOf(':');

    // A Windows drive letter also carries a colon, so only a parsable tail counts as a threshold.
    if (separator > 1
        && float.TryParse(argument[(separator + 1)..], NumberStyles.Float, CultureInfo.InvariantCulture, out var threshold))
    {
        paths.Add(argument[..separator]);
        thresholds.Add(threshold);
    }
    else
    {
        paths.Add(argument);
        thresholds.Add(0.5f);
    }
}

short[] samples;
try
{
    samples = WavFile.Read(recording);
}
catch (Exception e) when (e is IOException or InvalidDataException)
{
    Console.Error.WriteLine($"cannot read {recording}: {e.Message}");
    return 1;
}

using var detector = new WakeWordDetector(modelDirectory, paths, thresholds);

var names = paths.Select(Path.GetFileNameWithoutExtension).ToArray();
Console.WriteLine($"{recording}: {samples.Length / (double)WavFile.SampleRate:F1} s, "
    + $"listening for {string.Join(", ", names)}");

var detections = 0;
var frames = 0;
var peaks = new float[paths.Count];

for (var offset = 0; offset + WakeWordDetector.FrameLength <= samples.Length;
     offset += WakeWordDetector.FrameLength)
{
    var hit = detector.Process(samples.AsSpan(offset, WakeWordDetector.FrameLength));
    frames++;

    for (var i = 0; i < peaks.Length; i++)
    {
        peaks[i] = Math.Max(peaks[i], detector.LastScores[i]);
    }

    if (hit < 0)
    {
        continue;
    }

    detections++;
    var at = offset / (double)WavFile.SampleRate;
    Console.WriteLine($"  {at,6:F2} s  {names[hit]}  (score {detector.LastScores[hit]:F4})");
}

Console.WriteLine($"{frames} frames, {detections} detection(s)");

// Peak scores make it obvious whether a miss was close or nowhere near, which is what a threshold
// is tuned against.
for (var i = 0; i < peaks.Length; i++)
{
    Console.WriteLine($"  highest score for {names[i]}: {peaks[i]:F4} (threshold {thresholds[i]:F2})");
}

return 0;
