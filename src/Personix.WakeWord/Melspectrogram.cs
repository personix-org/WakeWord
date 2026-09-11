namespace Personix.WakeWord;

/// <summary>
/// Log-mel spectrogram with the parameters the speech embedding model was trained on.
///
/// <code>
/// 16 kHz PCM → Hann window of 400 samples in a 512-point FFT, hop 160
///            → power spectrum → 32 mel bands from 60 Hz to 3800 Hz (Slaney scale, area-normalised)
///            → 10·log10, floored at 80 dB below the loudest value of the call
/// </code>
///
/// This reproduces the <c>melspectrogram.onnx</c> that ships with openWakeWord to float precision,
/// so classifiers trained through that model keep their scores. It exists as code rather than a
/// model file because the transform is a fixed function with no trained weights in it.
///
/// One instance belongs to one thread — the FFT works in buffers shared between calls.
/// </summary>
public sealed class Melspectrogram
{
    /// <summary>Input sample rate.</summary>
    public const int SampleRate = 16000;

    /// <summary>Points in the FFT, and the length of every analysis frame.</summary>
    public const int FftSize = 512;

    /// <summary>Samples under the Hann window — 25 ms — centred in the FFT frame.</summary>
    public const int WindowLength = 400;

    /// <summary>Samples between consecutive frames — 10 ms.</summary>
    public const int HopLength = 160;

    /// <summary>Mel bands in the output.</summary>
    public const int MelBins = 32;

    private const int SpectrumBins = (FftSize / 2) + 1;
    private const double MinFrequency = 60.0;
    private const double MaxFrequency = 3800.0;

    /// <summary>Floor under the power before the logarithm — the <c>amin</c> of librosa.</summary>
    private const double PowerFloor = 1e-10;

    /// <summary>The output never drops further than this below its loudest value.</summary>
    private const double DynamicRangeDb = 80.0;

    private readonly double[] _window = HannWindow();
    private readonly MelFilter[] _melFilters = MelFilterBank();

    private readonly double[] _real = new double[FftSize];
    private readonly double[] _imaginary = new double[FftSize];
    private readonly double[] _cosine = new double[FftSize / 2];
    private readonly double[] _sine = new double[FftSize / 2];
    private readonly int[] _bitReversed = new int[FftSize];
    private readonly float[] _power = new float[SpectrumBins];

    public Melspectrogram()
    {
        for (var i = 0; i < FftSize / 2; i++)
        {
            _cosine[i] = Math.Cos(Math.Tau * i / FftSize);
            _sine[i] = Math.Sin(Math.Tau * i / FftSize);
        }

        var bits = int.Log2(FftSize);
        for (var i = 0; i < FftSize; i++)
        {
            var reversed = 0;
            for (var b = 0; b < bits; b++)
            {
                reversed = (reversed << 1) | ((i >> b) & 1);
            }

            _bitReversed[i] = reversed;
        }
    }

    /// <summary>How many frames <see cref="Compute"/> yields for this many samples.</summary>
    public static int FramesFor(int sampleCount) =>
        sampleCount < FftSize ? 0 : ((sampleCount - FftSize) / HopLength) + 1;

    /// <summary>
    /// Computes the log-mel frames of the samples into <paramref name="output"/>, frame after
    /// frame, <see cref="MelBins"/> values each. Returns the number of frames written.
    ///
    /// The 80 dB floor is relative to the loudest value of this call, so the same audio split
    /// differently gives slightly different numbers — the reference model behaves the same way.
    /// </summary>
    public int Compute(ReadOnlySpan<float> samples, Span<float> output)
    {
        var frames = FramesFor(samples.Length);

        if (output.Length < frames * MelBins)
        {
            throw new ArgumentException(
                $"Need room for {frames * MelBins} values, got {output.Length}.", nameof(output));
        }

        var loudest = double.NegativeInfinity;

        for (var f = 0; f < frames; f++)
        {
            PowerSpectrum(samples.Slice(f * HopLength, FftSize));

            var row = output.Slice(f * MelBins, MelBins);

            for (var m = 0; m < MelBins; m++)
            {
                var filter = _melFilters[m];
                var energy = 0.0;
                for (var i = 0; i < filter.Weights.Length; i++)
                {
                    energy += _power[filter.FirstBin + i] * filter.Weights[i];
                }

                var decibels = 10.0 * Math.Log10(Math.Max(energy, PowerFloor));
                loudest = Math.Max(loudest, decibels);
                row[m] = (float)decibels;
            }
        }

        var floor = (float)(loudest - DynamicRangeDb);
        var written = output[..(frames * MelBins)];

        for (var i = 0; i < written.Length; i++)
        {
            written[i] = Math.Max(written[i], floor);
        }

        return frames;
    }

    /// <summary>Windowed FFT of one frame; leaves |X|² of the non-negative frequencies in <see cref="_power"/>.</summary>
    private void PowerSpectrum(ReadOnlySpan<float> frame)
    {
        for (var i = 0; i < FftSize; i++)
        {
            _real[_bitReversed[i]] = frame[i] * _window[i];
            _imaginary[_bitReversed[i]] = 0.0;
        }

        // Iterative radix-2 Cooley–Tukey over the bit-reversed input.
        for (var size = 2; size <= FftSize; size <<= 1)
        {
            var half = size >> 1;
            var stride = FftSize / size;

            for (var start = 0; start < FftSize; start += size)
            {
                for (var k = 0; k < half; k++)
                {
                    var even = start + k;
                    var odd = even + half;
                    var cos = _cosine[k * stride];
                    var sin = _sine[k * stride];

                    var oddReal = (_real[odd] * cos) + (_imaginary[odd] * sin);
                    var oddImaginary = (_imaginary[odd] * cos) - (_real[odd] * sin);

                    _real[odd] = _real[even] - oddReal;
                    _imaginary[odd] = _imaginary[even] - oddImaginary;
                    _real[even] += oddReal;
                    _imaginary[even] += oddImaginary;
                }
            }
        }

        for (var bin = 0; bin < SpectrumBins; bin++)
        {
            _power[bin] = (float)((_real[bin] * _real[bin]) + (_imaginary[bin] * _imaginary[bin]));
        }
    }

    /// <summary>Periodic Hann window of <see cref="WindowLength"/> samples, zero-padded to the FFT size on both sides.</summary>
    private static double[] HannWindow()
    {
        var window = new double[FftSize];
        var offset = (FftSize - WindowLength) / 2;

        for (var n = 0; n < WindowLength; n++)
        {
            window[offset + n] = 0.5 - (0.5 * Math.Cos(Math.Tau * n / WindowLength));
        }

        return window;
    }

    /// <summary>One triangular filter: its weights, starting at <paramref name="FirstBin"/> of the spectrum.</summary>
    private readonly record struct MelFilter(int FirstBin, float[] Weights);

    /// <summary>
    /// Triangular mel filters as librosa builds them: Slaney's mel scale, each filter scaled to
    /// unit area. A filter touches a handful of the 257 spectrum bins and is zero everywhere
    /// else, so only the covered bins are kept — 229 weights instead of 8 224.
    /// </summary>
    private static MelFilter[] MelFilterBank()
    {
        var edges = new double[MelBins + 2];
        var lowest = HertzToMel(MinFrequency);
        var highest = HertzToMel(MaxFrequency);

        for (var i = 0; i < edges.Length; i++)
        {
            edges[i] = MelToHertz(lowest + ((highest - lowest) * i / (MelBins + 1)));
        }

        var filters = new MelFilter[MelBins];
        var column = new float[SpectrumBins];

        for (var m = 0; m < MelBins; m++)
        {
            var lower = edges[m];
            var centre = edges[m + 1];
            var upper = edges[m + 2];
            var scale = 2.0 / (upper - lower);
            var first = -1;
            var last = -1;

            for (var bin = 0; bin < SpectrumBins; bin++)
            {
                var hertz = bin * (SampleRate / 2.0) / (SpectrumBins - 1);
                var rising = (hertz - lower) / (centre - lower);
                var falling = (upper - hertz) / (upper - centre);
                var weight = Math.Max(0.0, Math.Min(rising, falling));

                column[bin] = (float)(weight * scale);

                if (column[bin] > 0f)
                {
                    first = first < 0 ? bin : first;
                    last = bin;
                }
            }

            filters[m] = new MelFilter(first, column[first..(last + 1)]);
        }

        return filters;
    }

    // Slaney's scale: linear below 1 kHz, logarithmic above, as in librosa.
    private const double LinearStep = 200.0 / 3.0;
    private const double LogBreakHertz = 1000.0;
    private const double LogBreakMel = LogBreakHertz / LinearStep;
    private static readonly double LogStep = Math.Log(6.4) / 27.0;

    private static double HertzToMel(double hertz) =>
        hertz < LogBreakHertz
            ? hertz / LinearStep
            : LogBreakMel + (Math.Log(hertz / LogBreakHertz) / LogStep);

    private static double MelToHertz(double mel) =>
        mel < LogBreakMel
            ? mel * LinearStep
            : LogBreakHertz * Math.Exp(LogStep * (mel - LogBreakMel));
}
