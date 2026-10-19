// True Envelope spectral envelope estimator (iterative cepstral smoothing,
// Imai & Abe; fast recipe after Roebel & Rodet DAFx'05) and the formant
// correction gain for resampler-decoupled pitch shifting.
//
// Speed: the envelope is smooth by definition, so estimation runs on a
// max-filter-subsampled spectrum (factor n/1024) with a 1024-point cepstrum
// (the Roebel/Rodet subsampling optimization) and is interpolated back to
// the full bin grid. ~8x cheaper than full-size cepstra, no measurable
// envelope quality change.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "pb_fft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace pbshift {

class TrueEnvelope {
public:
    // fftSize: analysis frame size (full bins = fftSize/2+1)
    // subFactor default 1: full-resolution cepstra. Subsampling (Roebel
    // max-filter trick) measured +1..+2.5 dB envelope LSD on speech — kept
    // available for a future low-CPU tier only.
    TrueEnvelope(int fftSize, int order, int subFactor = 0, int sampleRate = 48000)
        : nFull_(fftSize),
          binsFull_(fftSize / 2 + 1),
          sub_(subFactor > 0 ? subFactor : 1),
          n_(fftSize / sub_),
          bins_(n_ / 2 + 1),
          order_(order),
          baseOrder_(order),
          sampleRate_(sampleRate),
          fft_(n_),
          acfFft_(fftSize) {
        logS_.resize(bins_);
        env_.resize(bins_);
        ceps_ = RealFFT::alloc(n_);
        acf_ = RealFFT::alloc(fftSize);
        spec_.resize(bins_);
        aspec_.resize(binsFull_);
    }
    ~TrueEnvelope() {
        pffft_aligned_free(ceps_);
        pffft_aligned_free(acf_);
    }
    TrueEnvelope(const TrueEnvelope&) = delete;
    TrueEnvelope& operator=(const TrueEnvelope&) = delete;

    // Fundamental bin (full grid) detected during the last compute(); 0 if the
    // frame was not confidently voiced. Callers use it to leave the region
    // below F0 uncorrected.
    double lastF0Bin() const { return lastF0Bin_; }

    // Per-stream state (the smoothed F0 period that caps the cepstral order):
    // a new stream must not inherit the previous one's.
    void reset() {
        smQuef_ = 0.0;
        lastF0Bin_ = 0.0;
    }

    // F0 search range of the signal the envelope is computed on (the engine
    // moves it with the pre-resampler on pitch-down).
    void setF0Range(double minHz, double maxHz) {
        f0MinHz_ = minHz;
        f0MaxHz_ = maxHz;
    }

    // mag[binsFull] -> envLog[binsFull] (natural-log envelope)
    // orderOverride: cepstral order in FULL-grid units (scaled internally).
    void compute(const float* mag, std::vector<float>& envLog,
                 int orderOverride = 0) {
        const int reqOrder = orderOverride > 0 ? orderOverride : baseOrder_;
        order_ = std::min(std::max(4, reqOrder / sub_), n_ / 2 - 8);

        // max-filter subsampling onto the small grid
        float maxM = 1e-12f;
        for (int m = 0; m < binsFull_; ++m) maxM = std::max(maxM, mag[m]);
        const float floorLog = std::log(maxM) - 23.0f;  // -200 dB rel
        for (int q = 0; q < bins_; ++q) {
            const int lo = q * sub_ - sub_ / 2;
            const int hi = lo + sub_;
            float mx = 0.0f;
            for (int m = std::max(0, lo); m < std::min(binsFull_, hi); ++m)
                mx = std::max(mx, mag[m]);
            logS_[q] = std::max(std::log(mx + 1e-20f), floorLog);
        }

        // CRITICAL: the cepstral order MUST stay below the fundamental's
        // period, or the envelope tracks the harmonics themselves. When that
        // happens, the formant "correction" re-imposes the INPUT's harmonic
        // structure on the pitch-shifted output and silently undoes the pitch
        // shift (measured: sung vowel +12 st f0 unchanged). Downward capping
        // is always safe, so no confidence gate is needed.
        {
            const double lag = detectF0Lag(mag);
            // Temporal smoothing (order stability on vibrato) in the log
            // domain, restarted on a jump: averaging across an octave error
            // or a new note produced a period that belongs to neither (the old
            // linear EMA of the cepstral pick sat between T0 and 2T0).
            if (lag > 0.0) {
                smQuef_ = (smQuef_ > 0.0 && std::abs(std::log(lag / smQuef_)) < 0.12)
                              ? std::exp(0.7 * std::log(smQuef_) + 0.3 * std::log(lag))
                              : lag;
                // 0.5x the F0 period (the shorter of smoothed and current):
                // the envelope period stays >= 2x the fundamental period, so
                // env(f*p)/env(f) carries no harmonic ripple. Looser caps
                // (0.7x) leave residual ripple that couples harmonic k to 2k
                // and grows an f0/2 subharmonic on up-shifts (measured: +7 st
                // fund-to-subharmonic gap 14->25 dB).
                const double T = std::min(lag, smQuef_);
                const int cap = std::max(6, static_cast<int>(T / (2.0 * sub_)));
                if (cap < order_) order_ = cap;
                lastF0Bin_ = nFull_ / lag;   // full-grid fundamental bin
            } else {
                lastF0Bin_ = 0.0;
            }
        }

        std::copy(logS_.begin(), logS_.end(), env_.begin());
        const float stopLn = 2.0f * 0.115129f;  // 2 dB in ln units
        for (int it = 0; it < 8; ++it) {
            lifter(env_.data());
            float worst = 0.0f;
            for (int q = 0; q < bins_; ++q) {
                const float d = logS_[q] - env_[q];
                if (d > worst) worst = d;
            }
            if (worst < stopLn) break;
            for (int q = 0; q < bins_; ++q)
                env_[q] = std::max(env_[q], logS_[q]);
        }

        // interpolate back to the full grid
        envLog.resize(binsFull_);
        const double scale = static_cast<double>(bins_ - 1) / (binsFull_ - 1);
        for (int m = 0; m < binsFull_; ++m) {
            const double q = m * scale;
            const int q0 = static_cast<int>(q);
            const float fr = static_cast<float>(q - q0);
            envLog[m] = q0 + 1 < bins_
                            ? env_[q0] * (1.0f - fr) + env_[q0 + 1] * fr
                            : env_[bins_ - 1];
        }
    }

private:
    // Fundamental period (full-rate samples, fractional; 0 = unvoiced) from
    // the frame's power spectrum: autocorrelation = IFFT(|X|^2), divided by
    // the analysis window's own autocorrelation (Boersma 1993, the Praat
    // pitch method). Periodicity in the WAVEFORM is far more robust than the
    // cepstral peak on voices whose spectrum is one strong fundamental over a
    // breathy floor (sung vowel: the cepstral pick jumped between T0, 2T0,
    // 3T0 and T0/2, frame to frame). Octave guard: the SHORTEST lag whose
    // normalized correlation is within 85% of the best wins, so sub-octave
    // (2T0) errors -- the harmful direction for both the order cap and the
    // sub-F0 protection -- are avoided.
    double detectF0Lag(const float* mag) {
        for (int m = 0; m < binsFull_; ++m)
            aspec_[m] = {mag[m] * mag[m], 0.0f};
        acfFft_.inverse(aspec_.data(), acf_);
        const double r0 = acf_[0];
        if (!(r0 > 1e-20)) return 0.0;
        // circular ACF: keep lags <= N/3 where the wrapped term is < 6%
        const int lo = std::max(2, static_cast<int>(sampleRate_ / f0MaxHz_));
        const int hi = std::min(nFull_ / 3,
                                static_cast<int>(sampleRate_ / f0MinHz_) + 1);
        if (hi <= lo + 2) return 0.0;
        // periodic-Hann window autocorrelation (analytic, normalized)
        auto rw = [&](double t) {
            const double x = t / nFull_;
            return (1.0 - x) * (2.0 / 3.0 + std::cos(2.0 * M_PI * x) / 3.0) +
                   std::sin(2.0 * M_PI * x) / (2.0 * M_PI);
        };
        auto nr = [&](int t) { return acf_[t] / (r0 * rw(t)); };
        double best = 0.0;
        for (int t = lo; t <= hi; ++t) {
            const double v = nr(t);
            if (v > best && v >= nr(t - 1) && v >= nr(t + 1)) best = v;
        }
        if (best < 0.45) return 0.0;   // Praat's voicing threshold
        for (int t = lo; t <= hi; ++t) {
            const double v = nr(t), a = nr(t - 1), b = nr(t + 1);
            if (v >= 0.85 * best && v >= a && v >= b) {
                const double den = a - 2.0 * v + b;
                const double d = den < 0.0 ? 0.5 * (a - b) / den : 0.0;
                return t + std::clamp(d, -0.5, 0.5);
            }
        }
        return 0.0;
    }

    // cepstral low-pass: keep quefrency bins [0, order_]
    void lifter(float* logSpec) {
        for (int q = 0; q < bins_; ++q) spec_[q] = {logSpec[q], 0.0f};
        fft_.inverse(spec_.data(), ceps_);
        const int P = order_;
        for (int q = P + 1; q <= n_ - P - 1; ++q) ceps_[q] = 0.0f;
        for (int k = 0; k < 8 && P - k > 0; ++k) {
            const float w = 0.5f + 0.5f * std::cos(M_PI * (8 - k) / 8.0f);
            ceps_[P - k] *= 1.0f - (1.0f - w) * 0.5f;
            ceps_[n_ - (P - k)] *= 1.0f - (1.0f - w) * 0.5f;
        }
        fft_.forward(ceps_, spec_.data());
        for (int q = 0; q < bins_; ++q) logSpec[q] = spec_[q].real();
    }

    int nFull_, binsFull_, sub_, n_, bins_, order_, baseOrder_, sampleRate_;
    double lastF0Bin_ = 0.0;
    double smQuef_ = 0.0;  // log-EMA-smoothed F0 period (order stability)
    double f0MinHz_ = 60.0, f0MaxHz_ = 1100.0;
    RealFFT fft_;
    RealFFT acfFft_;
    std::vector<float> logS_, env_;
    std::vector<std::complex<float>> spec_, aspec_;
    float* ceps_;
    float* acf_;
};

// Formant corrector: desired(f) = env_in(f * p), gain clamped to +15/-30 dB.
// strength in [0,1] scales the correction: full correction is measured best
// for upshift and mild downshift, but over-corrects (artifacts) at deep
// downshift where partial/no correction scores higher perceived quality.
// The region below the fundamental has no envelope to restore; the caller
// (Stretcher stageFormant) leaves it uncorrected.
inline void formantGains(const std::vector<float>& envLog, double p,
                         std::vector<float>& gain, float strength = 1.0f) {
    const int B = static_cast<int>(envLog.size());
    gain.resize(B);
    // Asymmetric clamp: cuts are safe (they only attenuate existing content),
    // but large BOOSTS amplify floor/noise into spurious tones. Cap boost at
    // +15 dB, allow cut to -30 dB. +9 dB could not lift the envelope back up
    // at -12 st (the region above the moved-down formants sits >10 dB low):
    // speech formants stayed a full octave down (pbACC FR-FORMANT -1220 c),
    // +15 dB restores them (-23 c) without raising NISQA noisiness.
    const float limUp = 1.727f;    // ln(15 dB)
    const float limDn = -3.4539f;  // ln(30 dB)
    for (int m = 0; m < B; ++m) {
        const double q = m * p;
        const int q0 = static_cast<int>(q);
        float desired;
        if (q0 + 1 < B) {
            const float fr = static_cast<float>(q - q0);
            desired = envLog[q0] * (1.0f - fr) + envLog[q0 + 1] * fr;
        } else {
            desired = envLog[B - 1];  // flat extension above mapped Nyquist
        }
        float g = (desired - envLog[m]) * strength;
        g = std::clamp(g, limDn, limUp);
        gain[m] = std::exp(g);
    }
}

}  // namespace pbshift
