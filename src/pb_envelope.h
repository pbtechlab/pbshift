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
          fft_(n_) {
        logS_.resize(bins_);
        env_.resize(bins_);
        ceps_ = RealFFT::alloc(n_);
        spec_.resize(bins_);
    }
    ~TrueEnvelope() { pffft_aligned_free(ceps_); }
    TrueEnvelope(const TrueEnvelope&) = delete;
    TrueEnvelope& operator=(const TrueEnvelope&) = delete;

    // Fundamental bin (full grid) detected during the last compute(); 0 if the
    // frame was not confidently voiced. Callers use it to disable formant
    // correction below F0 (see formantGains).
    double lastF0Bin() const { return lastF0Bin_; }

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
        // quefrency, or the envelope tracks the harmonics themselves. When
        // that happens, the formant "correction" re-imposes the INPUT's
        // harmonic structure on the pitch-shifted output and silently undoes
        // the pitch shift (measured: sung vowel +12 st f0 unchanged). Cap the
        // order at 0.7x the detected F0 quefrency (harmonic-safe). Downward
        // capping is always safe, so no confidence gate is needed.
        {
            int q0 = detectF0Quefrency(logS_.data());
            // temporally smooth the quefrency so the order (and hence the
            // envelope) does not jitter frame-to-frame on vibrato/voiced
            // transitions — jitter re-introduces the subharmonic ripple.
            if (q0 > 0) {
                smQuef_ = smQuef_ > 0.0 ? 0.85 * smQuef_ + 0.15 * q0 : q0;
                q0 = static_cast<int>(smQuef_ + 0.5);
            }
            if (q0 > 0) {
                // 0.5x the F0 quefrency: the envelope period stays >= 2x the
                // fundamental period, so env(f*p)/env(f) carries no harmonic
                // ripple. Looser caps (0.7x) leave residual ripple that
                // couples harmonic k to 2k and grows an f0/2 subharmonic on
                // up-shifts (measured: +7 st fund-to-subharmonic gap 14->25 dB).
                const int cap = std::max(6, q0 / 2);
                if (cap < order_) order_ = cap;
                // full-grid fundamental bin = fftSize / quefrency(full-rate)
                lastF0Bin_ = static_cast<double>(nFull_) / (q0 * sub_);
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
    // Cepstral fundamental quefrency (in sub-grid samples) from the log
    // spectrum, restricted to the voice range (F0 70-500 Hz). Returns 0 if no
    // clear periodicity (unvoiced / non-harmonic -> no cap needed).
    int detectF0Quefrency(const float* logSpec) {
        for (int q = 0; q < bins_; ++q) spec_[q] = {logSpec[q], 0.0f};
        fft_.inverse(spec_.data(), ceps_);   // real cepstrum (sub-grid rate)
        const double subRate = static_cast<double>(sampleRate_) / sub_;
        const int lo = std::max(2, static_cast<int>(subRate / 500.0));
        const int hi = std::min(n_ / 2 - 1, static_cast<int>(subRate / 70.0));
        int qpk = 0;
        float pk = 0.0f;
        double meanAbs = 0.0;
        for (int q = lo; q <= hi; ++q) {
            meanAbs += std::abs(ceps_[q]);
            if (ceps_[q] > pk) { pk = ceps_[q]; qpk = q; }
        }
        meanAbs /= std::max(1, hi - lo + 1);
        return (pk > 2.5 * meanAbs) ? qpk : 0;
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
    double smQuef_ = 0.0;  // EMA-smoothed F0 quefrency (order stability)
    RealFFT fft_;
    std::vector<float> logS_, env_;
    std::vector<std::complex<float>> spec_;
    float* ceps_;
};

// Formant corrector: desired(f) = env_in(f * p), gain clamped to +-30 dB.
// strength in [0,1] scales the correction: full correction is measured best
// for upshift and mild downshift, but over-corrects (artifacts) at deep
// downshift where partial/no correction scores higher perceived quality.
// f0Bin: fundamental bin of the (engine-internal) signal, or <=0 if unknown.
// Below the fundamental there is no meaningful envelope, so a naive
// desired=env(f*p) samples the strong signal region while the current value
// is the sub-F0 noise floor -> a huge upward gain that amplifies the floor
// into a spurious tone (measured: voice +12 st grew a phantom partial at the
// ORIGINAL pitch, masking the shift). Below f0 the gain is forced to 1.
inline void formantGains(const std::vector<float>& envLog, double p,
                         std::vector<float>& gain, float strength = 1.0f,
                         double f0Bin = 0.0) {
    const int B = static_cast<int>(envLog.size());
    gain.resize(B);
    // Asymmetric clamp: cuts are safe (they only attenuate existing content),
    // but large BOOSTS amplify floor/noise into spurious tones. Cap boost at
    // +9 dB, allow cut to -30 dB.
    const float limUp = 1.036f;    // ln(9 dB)
    const float limDn = -3.4539f;  // ln(30 dB)
    const int f0lo = f0Bin > 1.0 ? static_cast<int>(0.85 * f0Bin) : 0;
    for (int m = 0; m < B; ++m) {
        if (m < f0lo) { gain[m] = 1.0f; continue; }  // sub-F0: no correction
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
