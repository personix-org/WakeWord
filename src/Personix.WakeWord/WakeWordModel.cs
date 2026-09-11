namespace Personix.WakeWord;

/// <summary>One wake word to listen for.</summary>
/// <param name="Word">A name for it — what <see cref="WakeWordDetection.Word"/> reports.</param>
/// <param name="Path">The classifier file, <c>.wwc</c> or <c>.onnx</c>.</param>
/// <param name="Threshold">Score from which it counts as heard.</param>
public sealed record WakeWordModel(string Word, string Path, float Threshold);
