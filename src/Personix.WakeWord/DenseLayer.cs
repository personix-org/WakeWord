namespace Personix.WakeWord;

/// <summary>A fully connected layer. Weights are stored per row: <c>Weights[output][input]</c>.</summary>
/// <param name="Weights">Weight matrix, one row per output neuron.</param>
/// <param name="Biases">Bias of each output neuron.</param>
public sealed record DenseLayer(float[][] Weights, float[] Biases);
