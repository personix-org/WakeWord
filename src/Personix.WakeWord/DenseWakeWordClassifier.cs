namespace Personix.WakeWord;

/// <summary>A classifier from the trainer's own format.</summary>
internal sealed class DenseWakeWordClassifier(DenseClassifier classifier, int expectedInputSize)
    : IWakeWordClassifier
{
    private readonly DenseClassifier _classifier = expectedInputSize == classifier.InputSize
        ? classifier
        : throw new InvalidDataException(
            $"the model expects {classifier.InputSize} values, the chain supplies {expectedInputSize}");

    public int InputSize => _classifier.InputSize;

    public float Score(ReadOnlySpan<float> window) => _classifier.Score(window);

    public void Dispose()
    {
        // Nothing native, just arrays of weights.
    }
}
