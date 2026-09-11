using Shouldly;

using Xunit;

namespace Personix.WakeWord.Tests;

public class WakeWordOptionsTests
{
    [Fact]
    public void Words_are_kept_in_the_order_they_were_added()
    {
        // Arrange
        var options = new WakeWordOptions();

        // Act
        options.Add("first", "first.wwc", 0.9f).Add("second", "second.onnx");

        // Assert — the index a detection reports is this order
        options.Words.Select(w => w.Word).ShouldBe(["first", "second"]);
        options.Words[1].Threshold.ShouldBe(0.5f, "the library's own default");
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(1.1f)]
    [InlineData(float.NaN)]
    public void A_threshold_outside_zero_to_one_is_refused(float threshold)
    {
        var options = new WakeWordOptions();

        Action add = () => options.Add("word", "word.wwc", threshold);

        Should.Throw<ArgumentOutOfRangeException>(add);
    }

    [Theory]
    [InlineData("", "word.wwc")]
    [InlineData("word", " ")]
    public void A_blank_name_or_path_is_refused(string word, string path)
    {
        var options = new WakeWordOptions();

        Action add = () => options.Add(word, path);

        Should.Throw<ArgumentException>(add);
    }

    [Fact]
    public void The_cooldown_covers_the_frames_one_utterance_spans()
    {
        // A word said once scores high on several consecutive 80 ms frames; the default has to
        // outlast them without swallowing a second utterance.
        new WakeWordOptions().Cooldown.ShouldBe(TimeSpan.FromMilliseconds(800));
    }
}
