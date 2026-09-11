using Shouldly;

using Xunit;

namespace Personix.WakeWord.Tests;

/// <summary>
/// The wake word classifier in the trainer's own format.
///
/// The trained part of openWakeWord is a small head over a window of embeddings — three linear
/// layers, two ReLUs and a sigmoid. The melspectrogram and embedding underneath are pre-trained
/// files that never change.
/// </summary>
public class DenseClassifierTests
{
    /// <summary>A network whose result can be worked out on paper.</summary>
    private static DenseClassifier OneNeuron(float weight, float bias) =>
        new([new DenseLayer(Weights: [[weight]], Biases: [bias])]);

    [Fact]
    public void A_single_neuron_puts_its_output_through_the_sigmoid()
    {
        // Arrange
        var classifier = OneNeuron(weight: 2f, bias: 0f);

        // Act
        var score = classifier.Score([1f]);

        // Assert — sigmoid(2·1 + 0) = 0.8808
        score.ShouldBe(0.8808f, 0.0001f);
    }

    [Fact]
    public void The_bias_shifts_the_result()
    {
        // Arrange
        var classifier = OneNeuron(weight: 0f, bias: 0f);

        // Act
        var score = classifier.Score([5f]);

        // Assert — sigmoid(0) = 0.5, the input goes nowhere
        score.ShouldBe(0.5f, 0.0001f);
    }

    /// <summary>ReLU sits between layers, not at the end — the last layer is closed by a sigmoid.</summary>
    [Fact]
    public void Negative_values_between_layers_are_cut_off()
    {
        // Arrange — the first layer gives -3, ReLU makes it 0, the second one just passes it on
        var classifier = new DenseClassifier([
            new DenseLayer(Weights: [[-3f]], Biases: [0f]),
            new DenseLayer(Weights: [[1f]], Biases: [0f]),
        ]);

        // Act
        var score = classifier.Score([1f]);

        // Assert — sigmoid(0) = 0.5, because ReLU dropped the negative value
        score.ShouldBe(0.5f, 0.0001f);
    }

    [Fact]
    public void Inputs_are_summed_with_their_weights()
    {
        // Arrange
        var classifier = new DenseClassifier([
            new DenseLayer(Weights: [[1f, 2f]], Biases: [0.5f]),
        ]);

        // Act
        var score = classifier.Score([3f, 4f]);

        // Assert — sigmoid(1·3 + 2·4 + 0.5) = sigmoid(11.5)
        score.ShouldBe(0.99999f, 0.0001f);
    }

    /// <summary>
    /// The dot product runs on SIMD over whole vectors, so a layer whose width is not a multiple
    /// of the vector width has a tail that has to be summed one value at a time.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(7)]
    [InlineData(96)]
    [InlineData(1536)]
    public void A_layer_of_any_width_sums_every_input(int width)
    {
        // Arrange — every weight 1 and no bias, so the score is the sigmoid of the plain sum
        var weights = new float[width];
        Array.Fill(weights, 1f);

        var window = new float[width];
        Array.Fill(window, 0.5f);

        var classifier = new DenseClassifier([new DenseLayer(Weights: [weights], Biases: [0f])]);

        // Act
        var score = classifier.Score(window);

        // Assert
        var expected = 1f / (1f + MathF.Exp(-(width * 0.5f)));
        score.ShouldBe(expected, 0.0001f);
    }

    [Fact]
    public void A_window_of_the_wrong_size_is_refused()
    {
        // Arrange
        var classifier = OneNeuron(weight: 1f, bias: 0f);

        // Act
        Action wrongSize = () => classifier.Score([1f, 2f]);

        // Assert — the message has to name both sizes, so it is clear what did not meet what
        var error = Should.Throw<ArgumentException>(wrongSize);
        error.Message.ShouldContain("1");
        error.Message.ShouldContain("2");
    }

    [Fact]
    public void Saving_and_loading_gives_back_the_same_scores()
    {
        // Arrange
        var original = new DenseClassifier([
            new DenseLayer(Weights: [[0.5f, -1.5f], [2f, 0.25f]], Biases: [0.1f, -0.2f]),
            new DenseLayer(Weights: [[1f, -1f]], Biases: [0.3f]),
        ]);
        float[] window = [0.7f, -0.4f];
        var path = Path.Combine(Path.GetTempPath(), $"classifier-{Guid.NewGuid():N}.wwc");

        // Act
        original.Save(path);
        var loaded = DenseClassifier.Load(path);

        // Assert
        try
        {
            loaded.Score(window).ShouldBe(original.Score(window));
            loaded.InputSize.ShouldBe(original.InputSize);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void A_file_that_is_not_a_classifier_is_refused()
    {
        // Arrange
        var path = Path.Combine(Path.GetTempPath(), $"nonsense-{Guid.NewGuid():N}.wwc");
        File.WriteAllText(path, "this is definitely not a model");

        // Act
        var load = () => DenseClassifier.Load(path);

        // Assert
        try
        {
            Should.Throw<InvalidDataException>(load);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void A_truncated_file_is_refused()
    {
        // Arrange — a valid header promising layers the file does not contain
        var path = Path.Combine(Path.GetTempPath(), $"truncated-{Guid.NewGuid():N}.wwc");
        using (var writer = new BinaryWriter(File.Create(path)))
        {
            writer.Write(System.Text.Encoding.ASCII.GetBytes("WWCLS1"));
            writer.Write(1);      // one layer
            writer.Write(1536);   // inputs
            writer.Write(128);    // outputs, and then nothing
        }

        // Act
        var load = () => DenseClassifier.Load(path);

        // Assert
        try
        {
            Should.Throw<InvalidDataException>(load);
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>The input layer says how many values the embedding window carries: 16 frames of 96.</summary>
    [Fact]
    public void Input_size_comes_from_the_first_layer()
    {
        // Arrange & Act
        var classifier = new DenseClassifier([
            new DenseLayer(Weights: [[1f, 2f, 3f]], Biases: [0f]),
        ]);

        // Assert
        classifier.InputSize.ShouldBe(3);
    }

    [Fact]
    public void A_classifier_without_layers_is_refused()
    {
        // Act
        var build = () => new DenseClassifier([]);

        // Assert
        Should.Throw<ArgumentException>(build);
    }

    [Fact]
    public void A_last_layer_with_more_than_one_output_is_refused()
    {
        // Act — a wake word classifier answers with a single probability
        var build = () => new DenseClassifier([
            new DenseLayer(Weights: [[1f], [1f]], Biases: [0f, 0f]),
        ]);

        // Assert
        Should.Throw<ArgumentException>(build);
    }

    [Fact]
    public void Scoring_a_window_does_not_allocate()
    {
        // Arrange — the shape the chain really uses: 1536 → 128 → 64 → 1
        var classifier = new DenseClassifier([
            new DenseLayer(Weights: Rows(outputs: 128, inputs: 1536), Biases: new float[128]),
            new DenseLayer(Weights: Rows(outputs: 64, inputs: 128), Biases: new float[64]),
            new DenseLayer(Weights: Rows(outputs: 1, inputs: 64), Biases: new float[1]),
        ]);

        var window = new float[1536];
        for (var i = 0; i < window.Length; i++)
        {
            window[i] = i / 1536f;
        }

        for (var i = 0; i < 50; i++)
        {
            classifier.Score(window);   // warm the JIT up, so its work is not measured
        }

        // Act
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 100; i++)
        {
            classifier.Score(window);
        }

        var perCall = (GC.GetAllocatedBytesForCurrentThread() - before) / 100;

        // Assert — the classifier scores every frame for as long as the process runs
        perCall.ShouldBe(0);
    }

    private static float[][] Rows(int outputs, int inputs)
    {
        var random = new Random(1337);
        var rows = new float[outputs][];

        for (var o = 0; o < outputs; o++)
        {
            rows[o] = new float[inputs];
            for (var i = 0; i < inputs; i++)
            {
                rows[o][i] = (float)(random.NextDouble() - 0.5);
            }
        }

        return rows;
    }
}
