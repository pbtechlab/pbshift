// Multi-resolution time-stretch (R3-style): partition the spectrum into bands,
// process each band with a window sized for it -- a LONG window for the low band
// (fine frequency resolution keeps dense partials separated, killing the
// phase-vocoder "chorusing"/flutter), a SHORT window for the high band (fine
// time resolution keeps transients crisp) -- then sum the band outputs.
//
// Unlike a time-domain crossover (FIR split then stretch each band), the band
// split here is done in the STFT magnitude domain per scale, at the SAME frame
// timing, so the bands stay phase-aligned across the boundary. Boundaries are
// placed near spectral valleys (fixed Hz; R3 uses ~700 / 4800 Hz) to minimise
// energy at the cut and thus cross-band beating.
//
// Transient handling: on a broadband onset the local rate is pinned to 1.0 for
// the frames the attack spans (attack not time-spread), timing debt repaid over
// ~100 ms -- ported from the streaming engine's scheduler.
//
// Offline (whole-signal) processing, reusing AnalysisFrontend / Rtpghi /
// WolaSynth per scale. Time-stretch only (pitch is handled by the outer
// resampler, as in the streaming path). Multi-channel: a reference channel
// drives peak/phase/onset/pin decisions and the inter-channel phase difference
// is copied verbatim, matching the streaming engine's stereo lock.
#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <vector>

#include "pb_pghi.h"
#include "pb_stft.h"
#include "pb_window.h"

namespace pbshift {

class MultiResStretch {
public:
    struct Scale {
        int n;          // window / FFT size
        double loHz;    // band lower edge (inclusive)
        double hiHz;    // band upper edge (exclusive); >= Nyquist = to top
    };

    MultiResStretch(int sampleRate, std::vector<Scale> scales, bool pin = true,
                    int hopDiv = 4)
        : sr_(sampleRate), scales_(std::move(scales)), pin_(pin),
          hopDiv_(std::max(2, hopDiv)) {}

    // Multi-resolution layout: long window below ~800 Hz (frequency resolution
    // kills chorusing on dense mixes), short window above (transient time
    // resolution). The general-purpose default.
    static std::vector<Scale> defaultScales() {
        return {{16384, 0.0, 800.0}, {4096, 800.0, 1e12}};
    }

    // Single long window across the whole band, no pinning. Best for sustained,
    // strongly harmonic monophonic material (a held sung vowel): splitting dense
    // voice harmonics across a crossover decorrelates them and adds flutter,
    // whereas one long window keeps every harmonic phase-coherent, and such
    // material has no sharp attacks to protect. Select via Voice mode.
    //
    // The window is ratio-adaptive: under expansion (ratio > 1) a longer window
    // (32768) tracks the slowly-evolving harmonics better and wins the mild
    // up-stretch band (e.g. sung vowel 1.25x); under compression a 16384 window
    // is better (32768 over-smooths and loses at 0.5x).
    static std::vector<Scale> voicedScales(double ratio = 1.0) {
        return {{ratio > 1.0 ? 32768 : 16384, 0.0, 1e12}};
    }

    // Single moderate window, no crossover. For broadband percussive material
    // (drums, castanets): a band split spreads a broadband click's low-frequency
    // energy across the long low-band window and dulls the attack, so a single
    // window with NO crossover keeps impulses razor-sharp. Pinning still holds
    // each hit at rate 1.0.
    static std::vector<Scale> percussiveScales() {
        return {{4096, 0.0, 1e12}};
    }

    // Content-adaptive layout, chosen from whole-signal features:
    //  - broadband/noisy (high spectral flatness)  -> percussive single window
    //  - very tonal + sustained + monophonic        -> voiced single long window
    //  - otherwise (tonal mix, dense harmonic)       -> multi-resolution split
    // Robust: spectral flatness cleanly separates percussion (>~0.06) from tonal
    // (<~0.01); the tonal-vs-mix split falls through to multires, which is the
    // safe general default. Sets *pinOut (percussion/mix pin; voiced no pin).
    static std::vector<Scale> autoScales(const std::vector<float>& x, int sr,
                                         bool* pinOut = nullptr,
                                         double ratio = 1.0) {
        const double flat = spectralFlatness(x);
        const double per = periodicity(x, sr);
        // Threshold set ABOVE dense musical mixes (which carry drums but also
        // sustained tonal content that needs the long window) and below pure
        // percussion, so a full mix stays on the multi-resolution path.
        if (flat > 0.14) {                       // broadband percussion only
            if (pinOut) *pinOut = true;
            return percussiveScales();
        }
        if (flat < 0.002 && per > 0.85) {        // pure sustained tone / held vowel
            if (pinOut) *pinOut = false;
            return voicedScales(ratio);
        }
        if (pinOut) *pinOut = true;              // tonal mix / dense harmonic
        return defaultScales();
    }

    // Spectral flatness (geometric mean / arithmetic mean of the power
    // spectrum): ~0 for a pure tone, ->1 for white noise. Whole-signal.
    static double spectralFlatness(const std::vector<float>& x) {
        if (x.size() < 64) return 0.0;
        int n = 1;
        while (n * 2 <= (int)x.size() && n < 1 << 16) n <<= 1;
        WindowSet w = WindowSet::hann(n);
        RealFFT fft(n);
        float* buf = RealFFT::alloc(n);
        std::vector<std::complex<float>> X(fft.bins());
        for (int i = 0; i < n; ++i) buf[i] = x[i] * w.w[i];
        fft.forward(buf, X.data());
        pffft_aligned_free(buf);
        double logSum = 0.0, sum = 0.0;
        int cnt = 0;
        for (int m = 1; m < fft.bins(); ++m) {
            const double p = std::norm(X[m]) + 1e-12;
            logSum += std::log(p);
            sum += p;
            ++cnt;
        }
        if (cnt == 0 || sum <= 0) return 0.0;
        return std::exp(logSum / cnt) / (sum / cnt);
    }

    // Normalized pitch-autocorrelation peak (60-500 Hz) over a central window;
    // ~1 for a clean periodic tone/vowel, low for noise/inharmonic content.
    static double periodicity(const std::vector<float>& x, int sr) {
        const int W = std::min<int>((int)x.size(), sr / 5);   // 200 ms
        if (W < sr / 300) return 0.0;
        const size_t c = x.size() / 2;
        const size_t s = (c > (size_t)W / 2) ? c - W / 2 : 0;
        double z = 0.0;
        for (int i = 0; i < W; ++i) z += (double)x[s + i] * x[s + i];
        if (z < 1e-12) return 0.0;
        double best = 0.0;
        for (int lag = sr / 500; lag <= sr / 60 && lag < W; ++lag) {
            double r = 0.0;
            for (int i = 0; i + lag < W; ++i)
                r += (double)x[s + i] * x[s + i + lag];
            best = std::max(best, r / z);
        }
        return best;
    }

    // --- multi-channel API (reference-channel phase lock) -------------------
    std::vector<std::vector<float>> process(
        const std::vector<std::vector<float>>& in, double ratio) {
        const int C = (int)in.size();
        const long long nIn = C ? (long long)in[0].size() : 0;
        const long long nOut = llround(nIn * ratio);
        std::vector<std::vector<double>> acc(
            C, std::vector<double>((size_t)nOut + 1, 0.0));

        for (const Scale& s : scales_) {
            const int N = s.n;
            const int B = N / 2 + 1;
            // Under compression (ratio<1) the analysis hop = hs/ratio grows and
            // under-samples the phase evolution -> more flutter; raise the
            // synthesis overlap there to keep the analysis hop fine.
            const int hopDiv = ratio < 1.0 ? std::max(hopDiv_, 8) : hopDiv_;
            const int hs = N / hopDiv;                  // synthesis OLA overlap
            const double nyq = 0.5 * sr_;
            const int loBin = (int)std::floor(std::min(s.loHz, nyq) * N / sr_);
            const int hiBin = (int)std::ceil(std::min(s.hiHz, nyq) * N / sr_);
            const int lo = std::clamp(loBin, 0, B);
            const int hi = std::clamp(std::max(lo, hiBin), 0, B);

            const int pad = N / 2;
            std::vector<std::vector<float>> x(
                C, std::vector<float>((size_t)nIn + 2 * pad, 0.0f));
            for (int c = 0; c < C; ++c)
                for (long long i = 0; i < nIn; ++i) x[c][i + pad] = in[c][i];

            WindowSet win = WindowSet::hann(N);
            AnalysisFrontend fe(N, win);
            const long long cap = nOut + 4LL * N;
            std::vector<std::unique_ptr<WolaSynth>> wola;
            for (int c = 0; c < C; ++c)
                wola.emplace_back(new WolaSynth(
                    N, win, (int)std::min<long long>(cap, 1LL << 26)));
            Rtpghi pghi(B);                             // ref-channel phase
            std::vector<AnalysisFrame> frames(C);
            std::vector<float> synthPhase;
            std::vector<float> prevMag(B, 0.0f);
            std::vector<uint8_t> resetMask(B, 0);
            double prevFlux = 0.0;

            const double nominal = (double)hs / ratio;
            const double framesPerRepay = std::max(1.0, 0.1 * sr_ / hs);
            double ia = 0.0, debt = 0.0;
            int pin = 0;
            for (long long k = 0;; ++k) {
                const long long ip = (long long)llround(ia);
                if (ip + N > (long long)x[0].size()) break;
                for (int c = 0; c < C; ++c) {
                    fe.analyze(x[c].data() + ip, frames[c]);
                    for (int m = 0; m < B; ++m)
                        if (m < lo || m >= hi) {
                            frames[c].mag[m] = 0.0f;
                            frames[c].X[m] = {0.0f, 0.0f};
                        }
                }
                const AnalysisFrame& R = frames[0];     // reference channel
                float maxMag = 0.0f;
                for (int m = lo; m < hi; ++m)
                    maxMag = std::max(maxMag, R.mag[m]);
                // Half-wave-rectified spectral flux + broadband rise fraction.
                // Only a BROADBAND rise (energy appearing across many bins) is a
                // real transient; a tonal vibrato / beating tone raises flux at
                // a few harmonic bins only. Gating the phase reset AND the pin to
                // a broadband, well-above-average flux spike stops spurious
                // firing on sustained/tonal material -- which otherwise injects
                // amplitude modulation (audible chorusing on a held tone) and a
                // fake transient into the timing (debt) machinery.
                double flux = 0.0;
                int risen = 0, nb = 0;
                for (int m = lo; m < hi; ++m) {
                    ++nb;
                    const float d = R.mag[m] - prevMag[m];
                    if (d > 0) {
                        flux += d;
                        if (d > 0.02f * maxMag) ++risen;
                    }
                    prevMag[m] = R.mag[m];
                }
                const double riseFrac = nb ? (double)risen / nb : 0.0;
                const bool transient = k > 0 && flux > 2.5 * prevFlux + 1e-9 &&
                                       riseFrac > 0.15;
                prevFlux = 0.5 * prevFlux + 0.5 * flux;
                if (pin_ && transient && std::abs(debt) < 4.0 * hs)
                    pin = N / hs;
                std::fill(resetMask.begin(), resetMask.end(),
                          transient ? (uint8_t)1 : (uint8_t)0);
                pghi.stepLocked(R, hs, transient ? resetMask.data() : nullptr,
                                synthPhase, nullptr, 0);

                const long long center = k * (long long)hs;
                for (int c = 0; c < C; ++c) {
                    const AnalysisFrame& F = frames[c];
                    Y_.assign(B, {0.0f, 0.0f});
                    if (c == 0) {
                        for (int m = lo; m < hi; ++m)
                            Y_[m] = std::polar(F.mag[m], synthPhase[m]);
                    } else {
                        for (int m = lo; m < hi; ++m) {
                            std::complex<float> rel = F.X[m] * std::conj(R.X[m]);
                            const float rm = std::abs(rel);
                            rel = rm > 1e-20f ? rel / rm
                                              : std::complex<float>(1.0f, 0.0f);
                            Y_[m] = F.mag[m] * std::polar(1.0f, synthPhase[m]) *
                                    rel;
                        }
                    }
                    if (lo == 0) Y_[0] = {std::abs(Y_[0]), 0.0f};
                    if (hi == B) Y_[B - 1] = {std::abs(Y_[B - 1]), 0.0f};
                    wola[c]->addFrame(Y_.data(), center);
                }

                double advance;
                if (pin > 0) {
                    advance = hs;
                    debt += advance - nominal;
                    --pin;
                } else if (std::abs(debt) > 1e-6) {
                    double repay = std::clamp(debt / framesPerRepay,
                                              -0.5 * nominal, 0.5 * nominal);
                    advance = nominal - repay;
                    debt -= repay;
                } else {
                    advance = nominal;
                }
                ia += advance;
            }
            std::vector<float> tmp((size_t)nOut, 0.0f);
            for (int c = 0; c < C; ++c) {
                wola[c]->read(0, (int)std::min<long long>(nOut, 1LL << 30),
                              tmp.data());
                for (long long i = 0; i < nOut; ++i) acc[c][i] += tmp[i];
            }
        }

        std::vector<std::vector<float>> out(C, std::vector<float>((size_t)nOut));
        for (int c = 0; c < C; ++c)
            for (long long i = 0; i < nOut; ++i)
                out[c][i] = (float)acc[c][i];
        return out;
    }

    // mono convenience wrapper
    std::vector<float> process(const std::vector<float>& in, double ratio) {
        return process(std::vector<std::vector<float>>{in}, ratio)[0];
    }

private:
    int sr_;
    std::vector<Scale> scales_;
    bool pin_ = true;
    int hopDiv_ = 4;
    std::vector<std::complex<float>> Y_;
};

}  // namespace pbshift
