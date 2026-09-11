using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace Personix.WakeWord;

/// <summary>Registers wake word detection as a hosted service.</summary>
public static class WakeWordServiceCollectionExtensions
{
    /// <summary>
    /// Adds the listener as a hosted service. Continue with the returned builder to give it an
    /// audio source and handlers.
    /// </summary>
    /// <example>
    /// <code>
    /// services.AddWakeWord(words => words
    ///         .Add("my-word", "models/my-word.wwc", threshold: 0.9f))
    ///     .AddAudioSource&lt;MicrophoneSource&gt;()
    ///     .AddHandler&lt;StartVoicePipeline&gt;();
    /// </code>
    /// </example>
    public static WakeWordBuilder AddWakeWord(this IServiceCollection services, Action<WakeWordOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);

        services.Configure(configure);
        services.AddSingleton<WakeWordListener>();
        services.AddSingleton<IHostedService>(provider => provider.GetRequiredService<WakeWordListener>());

        return new WakeWordBuilder(services);
    }
}
