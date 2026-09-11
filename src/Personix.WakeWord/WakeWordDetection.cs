namespace Personix.WakeWord;

/// <summary>One wake word heard.</summary>
/// <param name="Word">The word's name, as registered in <see cref="WakeWordOptions"/>.</param>
/// <param name="Index">The word's position in the registration order.</param>
/// <param name="Score">The classifier's score, 0 to 1, at the frame that fired.</param>
/// <param name="At">Position in the stream, counted in frames since the listener started.</param>
public sealed record WakeWordDetection(string Word, int Index, float Score, TimeSpan At);
