using Microsoft.Extensions.DependencyInjection;

namespace Personix.WakeWord;

/// <summary>Continues the registration started by <see cref="WakeWordServiceCollectionExtensions.AddWakeWord"/>.</summary>
public sealed class WakeWordBuilder(IServiceCollection services)
{
    /// <summary>The collection being registered into.</summary>
    public IServiceCollection Services { get; } = services;

    /// <summary>Registers the audio source the listener reads from. One per application.</summary>
    public WakeWordBuilder AddAudioSource<TSource>()
        where TSource : class, IAudioSource
    {
        Services.AddSingleton<IAudioSource, TSource>();
        return this;
    }

    /// <summary>Registers an audio source built by the factory. One per application.</summary>
    public WakeWordBuilder AddAudioSource(Func<IServiceProvider, IAudioSource> factory)
    {
        ArgumentNullException.ThrowIfNull(factory);
        Services.AddSingleton(factory);
        return this;
    }

    /// <summary>
    /// Registers something to run when a word is heard. Handlers are scoped: each detection gets
    /// a fresh scope, so a handler can take scoped services such as a database context.
    /// </summary>
    public WakeWordBuilder AddHandler<THandler>()
        where THandler : class, IWakeWordHandler
    {
        Services.AddScoped<IWakeWordHandler, THandler>();
        return this;
    }
}
