using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;

using Shouldly;

using Xunit;

namespace Personix.WakeWord.Tests;

public class WakeWordServiceCollectionExtensionsTests
{
    [Fact]
    public void AddWakeWord_registers_the_listener_as_a_hosted_service()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging();

        // Act
        services.AddWakeWord(words => words.Add("word", "word.wwc", 0.9f))
            .AddAudioSource<SilentSource>()
            .AddHandler<CountingHandler>();

        using var provider = services.BuildServiceProvider();

        // Assert
        provider.GetServices<IHostedService>().ShouldHaveSingleItem().ShouldBeOfType<WakeWordListener>();
        provider.GetRequiredService<WakeWordListener>().ShouldBeSameAs(provider.GetServices<IHostedService>().Single());
        provider.GetRequiredService<IOptions<WakeWordOptions>>().Value.Words.ShouldHaveSingleItem().Word.ShouldBe("word");
        provider.GetRequiredService<IAudioSource>().ShouldBeOfType<SilentSource>();
    }

    [Fact]
    public void Handlers_are_scoped_and_resolve_in_registration_order()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging();
        services.AddWakeWord(words => words.Add("word", "word.wwc"))
            .AddAudioSource<SilentSource>()
            .AddHandler<CountingHandler>()
            .AddHandler<ThrowingHandler>();

        using var provider = services.BuildServiceProvider();

        // Act
        using var scope = provider.CreateScope();
        var handlers = scope.ServiceProvider.GetServices<IWakeWordHandler>().ToList();

        // Assert
        handlers.Select(h => h.GetType()).ShouldBe([typeof(CountingHandler), typeof(ThrowingHandler)]);
        scope.ServiceProvider.GetServices<IWakeWordHandler>().First().ShouldBeSameAs(handlers[0], "scoped: one instance per scope");
    }

    [Fact]
    public void A_handler_can_be_limited_to_some_words()
    {
        // Arrange
        var services = new ServiceCollection();
        services.AddLogging();
        services.AddWakeWord(words => words.Add("one", "one.wwc").Add("two", "two.wwc"))
            .AddAudioSource<SilentSource>()
            .AddHandler<CountingHandler>("one");

        using var provider = services.BuildServiceProvider();
        using var scope = provider.CreateScope();

        // Act
        var handler = scope.ServiceProvider.GetServices<IWakeWordHandler>().ShouldHaveSingleItem();

        // Assert — wrapped in the filter, and the filter names what was registered
        handler.ShouldBeOfType<WordFilteredHandler>();
        handler.ToString().ShouldBe(nameof(CountingHandler));
        scope.ServiceProvider.GetRequiredService<CountingHandler>().ShouldNotBeNull("the handler itself is resolvable too");
    }

    [Fact]
    public void A_blank_word_on_a_handler_is_refused()
    {
        var builder = new ServiceCollection().AddWakeWord(words => words.Add("one", "one.wwc"));

        Action add = () => builder.AddHandler<CountingHandler>("one", " ");

        Should.Throw<ArgumentException>(add);
    }

    [Fact]
    public void A_factory_can_supply_the_audio_source()
    {
        var services = new ServiceCollection();
        services.AddLogging();
        var source = new SilentSource();

        services.AddWakeWord(words => words.Add("word", "word.wwc")).AddAudioSource(_ => source);

        using var provider = services.BuildServiceProvider();
        provider.GetRequiredService<IAudioSource>().ShouldBeSameAs(source);
    }

    private sealed class SilentSource : IAudioSource
    {
        public ValueTask<bool> ReadFrameAsync(Memory<short> frame, CancellationToken cancellationToken) =>
            ValueTask.FromResult(false);
    }

    private sealed class CountingHandler : IWakeWordHandler
    {
        public Task HandleAsync(WakeWordDetection detection, CancellationToken cancellationToken) => Task.CompletedTask;
    }

    private sealed class ThrowingHandler : IWakeWordHandler
    {
        public Task HandleAsync(WakeWordDetection detection, CancellationToken cancellationToken) =>
            throw new InvalidOperationException("boom");
    }
}
