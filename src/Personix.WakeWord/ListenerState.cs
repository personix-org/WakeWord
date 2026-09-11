namespace Personix.WakeWord;

/// <summary>What the <see cref="WakeWordListener"/> is doing right now.</summary>
public enum ListenerState
{
    /// <summary>Not running — before start, or after the source ended or the host stopped it.</summary>
    Stopped,

    /// <summary>Reading audio and detecting.</summary>
    Listening,

    /// <summary>Paused from outside: not reading, the source told to let the microphone go.</summary>
    Paused,
}
