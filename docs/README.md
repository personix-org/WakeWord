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
| `WakeWordDetector` | The whole chain behind one call. Feed it frames, it says which word fired. |
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

3. Feed the detector consecutive frames.

   ```csharp
   using Personix.WakeWord;

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
seconds it always returns -1, while the classifier window fills.

`LastScores` holds the score of every model from the last frame, which is what a threshold is
tuned against — a word that fires on similar-sounding speech wants a higher one than the 0.5 the
library suggests.

### Several words at once

The expensive part of the chain runs once per frame regardless of how many words are listened for.
Each further word is one pass over a small network:

```csharp
using var detector = new WakeWordDetector(
    null,
    ["hey_jarvis_v0.1.onnx", "alexa_v0.1.onnx", "my-word.wwc"],
    [0.5f, 0.5f, 0.9f]);
```

The index returned by `Process` says which one it was, in the order the models were given.

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
samples/Personix.WakeWord.Sample/ runs the detector over a WAV file
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
