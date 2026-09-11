# Personix.WakeWord

On-device wake word detection for .NET. No account, no access key, no network call.

Runs the [openWakeWord](https://github.com/dscripka/openWakeWord) model chain — melspectrogram,
speech embedding, classifier. The first two stages are shared by every wake word, so adding a
second or a tenth word costs one small classifier rather than another pipeline.

The chain is built to run for as long as the process lives: the melspectrogram is code, the
embedding model runs incrementally — each 80 ms frame recomputes only the columns it touches,
not the whole 1.28 s window — and nothing in the per-frame path allocates. Three wake words cost
about 0.5 ms per frame on one core of an Apple M4.

Audio capture is left to the host: the detector takes frames of 16 kHz mono PCM.

## Contents

| Type | What it does |
|---|---|
| `AddWakeWord(…)` | Registers the listener as a hosted service — the way an application uses this. |
| `IAudioSource` | Where frames come from. The host implements it over its microphone library. |
| `IWakeWordHandler` | What happens when a word is heard. Scoped, async, as many as you like. |
| `WakeWordOptions` | Which words, from where, with what threshold; the cooldown after a detection. |
| `WakeWordListener` | The hosted service: source → detector → handlers. |
| `WakeWordDetector` | The chain behind one synchronous call, for hosts with their own audio loop. |
| `WakeWordFeatures` | The shared half — audio to windows of speech embeddings. |
| `Melspectrogram` | The log-mel front end, as code. |
| `IWakeWordClassifier` | The trained head that scores one window. |
| `WakeWordClassifiers` | Opens a classifier, picking the reader by file extension. |
| `DenseClassifier` | A classifier in the `.wwc` format, and the reader and writer for it. |

## Usage

1. Install the package.

   ```sh
   dotnet add package Personix.WakeWord
   ```

2. Get a classifier for each word you want to hear — one you trained, or one of the pre-trained
   ones from openWakeWord (see [Models and licensing](#models-and-licensing) for their terms).
   The shared embedding model comes with the package.

3. Register the listener, give it an audio source and say what should happen.

   ```csharp
   using Personix.WakeWord;

   services.AddWakeWord(words => words
           .Add("my-word", "my-word.wwc", threshold: 0.9f))
       .AddAudioSource<MicrophoneSource>()
       .AddHandler<StartVoicePipeline>();
   ```

   The audio source is yours — the package has no opinion on how audio is captured:

   ```csharp
   sealed class MicrophoneSource(IMicrophone microphone) : IAudioSource
   {
       // Fill the frame with the next 1280 samples of 16 kHz mono PCM; false ends the listener.
       public ValueTask<bool> ReadFrameAsync(Memory<short> frame, CancellationToken ct)
           => microphone.ReadAsync(frame, ct);
   }
   ```

   Handlers are scoped services, so they can take whatever the detection needs:

   ```csharp
   sealed class StartVoicePipeline(IHttpClientFactory http) : IWakeWordHandler
   {
       public Task HandleAsync(WakeWordDetection detection, CancellationToken ct)
           => http.CreateClient("pipeline").PostAsJsonAsync("/voice/start", detection, ct);
   }
   ```

   A `WakeWordDetection` carries the word's name and index, the score, and its position in
   the stream. Handlers run in registration order; one that throws is logged and skipped.

   A handler can answer to some words only — the others never reach it:

   ```csharp
   services.AddWakeWord(words => words
           .Add("assistant", "assistant.wwc", threshold: 0.9f)
           .Add("stop", "stop.wwc", threshold: 0.9f))
       .AddAudioSource<MicrophoneSource>()
       .AddHandler<StartVoicePipeline>("assistant")
       .AddHandler<StopEverything>("stop")
       .AddHandler<LogEveryDetection>();          // no words: all of them
   ```

`WakeWordOptions.Cooldown` (800 ms by default) folds the several frames one utterance fires on
into a single detection. A word that fires on similar-sounding speech wants a higher threshold
than the 0.5 the library suggests.

### Pausing from outside

`WakeWordListener.Pause()` stops reading and tells the audio source to let the microphone go —
for as long as another application needs it. `Resume()` starts again from a clean window, and
`State` says which it is. Both are safe to call from any thread, so a control endpoint or a
hotkey can drive them:

```csharp
app.MapPost("/pause", (WakeWordListener listener) => { listener.Pause(); return listener.State; });
app.MapPost("/resume", (WakeWordListener listener) => { listener.Resume(); return listener.State; });
app.MapGet("/state", (WakeWordListener listener) => listener.State);
```

A source over a microphone overrides `IAudioSource.Pause` to close the device and `Resume` to
reopen it; a source over a file has nothing to release and can leave both alone.

### Without a host

`WakeWordListener.RunAsync` is public, so a console application can drive it without hosting:

```csharp
await using var provider = services.BuildServiceProvider();
await provider.GetRequiredService<WakeWordListener>().RunAsync(cancellationToken);
```

### Your own audio loop

`WakeWordDetector` is the chain behind a single synchronous call, for a host that already has
one. It allocates nothing per frame and has no opinion on threads.

```csharp
using var detector = new WakeWordDetector(
    modelDirectory: null,                       // the model the package installed
    wakeWordModelPaths: ["my-word.wwc"],
    thresholds: [0.9f]);

while (recorder.TryReadFrame(out short[] frame))   // 1280 samples, 16 kHz mono
{
    var hit = detector.Process(frame);

    if (hit >= 0)
    {
        Console.WriteLine($"heard word {hit}, score {detector.LastScores[hit]:F3}");
    }
}
```

`Process` returns the index of the word that fired, or -1 for none. For roughly the first 1.3
seconds it always returns -1, while the classifier window fills. `LastScores` holds every
model's score from the last frame, which is what a threshold is tuned against.

### Several words at once

The expensive part of the chain runs once per frame regardless of how many words are listened for.
Each further word is one pass over a small network:

```csharp
services.AddWakeWord(words => words
    .Add("jarvis", "hey_jarvis_v0.1.onnx")
    .Add("alexa", "alexa_v0.1.onnx")
    .Add("mine", "my-word.wwc", threshold: 0.9f));
```

The detection names the word and carries its index, in the order the words were added.

### Threading

One detector belongs to one thread. It carries the audio heard so far, and its buffers are reused
between frames rather than reallocated.

Call `Reset()` between unrelated recordings, so the tail of one does not bleed into the next.

## Models and licensing

The package is self-contained. Installing it puts the embedding model next to your binaries as
`models/embedding_b0.onnx` … `embedding_b5.onnx` — the network split at its pooling layers so it
can run incrementally — and `WakeWordDetector` looks there when `modelDirectory` is null. The
log-mel front end is code, not a model file.

| Part | What it is | License |
|---|---|---|
| the library | this code | MIT |
| `Melspectrogram` | an independent implementation of a fixed signal-processing transform | MIT |
| `models/embedding_b*.onnx` | Google's speech embedding model, published as the TensorFlow Hub module `google/speech_embedding/1` (Lin et al., 2020); split into six blocks, weights untouched | Apache-2.0 — see `NOTICE` and `LICENSE-embedding_model` |

The ONNX conversion of the embedding model is the one distributed by
[openWakeWord](https://github.com/dscripka/openWakeWord), whose code is likewise Apache-2.0.

**Not included:** the pre-trained wake words that ship with openWakeWord (`hey_jarvis`, `alexa`,
…). openWakeWord licenses those CC BY-NC-SA 4.0 — non-commercial, share-alike — because of
datasets with unknown or restrictive licensing in their training data. They still work with this
detector if you supply them, under their own terms.

Classifiers you train yourself are yours. The `.wwc` format exists for exactly that: a trainer can
write one without leaving .NET, since TorchSharp cannot export to ONNX.

## Project structure

```
src/Personix.WakeWord/            the library, with build/Personix.WakeWord.targets that ships the model
models/                           embedding_b0…b5.onnx (Apache-2.0, see NOTICE)
tests/Personix.WakeWord.Tests/    unit tests, plus chain tests against reference recordings
samples/Personix.WakeWord.Sample/ the DI wiring over a WAV file source, printing detections
docs/README.md                    this file
```

## Development

```sh
dotnet build
dotnet test
```

The chain tests compare against reference numbers taken on real recordings and are skipped unless
the environment points at them:

```sh
export WAKEWORD_MODELS=/path/to/models       # the .wwc classifiers, plus openWakeWord's melspectrogram.onnx and embedding_model.onnx as references
export WAKEWORD_SAMPLES=/path/to/recordings  # one folder per word
dotnet test
```

Run the sample over a recording:

```sh
dotnet run --project samples/Personix.WakeWord.Sample -- \
    ./models ./recording.wav ./my-word.wwc:0.9
```

## License

MIT — see [LICENSE](../LICENSE). The bundled embedding model is Apache-2.0, see
[Models and licensing](#models-and-licensing) and [NOTICE](../NOTICE).
