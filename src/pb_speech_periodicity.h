// Window-corrected autocorrelation used only for speech-path eligibility.
#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "pb_fft.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
namespace pbshift {
class SpeechPeriodicity {
   public:
    SpeechPeriodicity(int size, int sampleRate)
        : nFull_(size), binsFull_(size / 2 + 1), sampleRate_(sampleRate), acfFft_(size) {
        aspec_.resize(binsFull_);
        acf_ = RealFFT::alloc(size);
    }
    ~SpeechPeriodicity() { pffft_aligned_free(acf_); }
    SpeechPeriodicity(const SpeechPeriodicity&) = delete;
    SpeechPeriodicity& operator=(const SpeechPeriodicity&) = delete;
    void reset() { lastPeriodicity_ = 0; }
    void setF0Range(double low, double high) {
        f0MinHz_ = low;
        f0MaxHz_ = high;
    }
    double lastPeriodicity() const { return lastPeriodicity_; }
    double fundamentalBin(const float* magnitude) {
        const double lag = detectF0Lag(magnitude, true);
        return lag > 0 ? nFull_ / lag : 0;
    }

   private:
    double detectF0Lag(const float* mag, bool persistentShort = false) {
        lastPeriodicity_ = 0.0;
        for (int m = 0; m < binsFull_; ++m) aspec_[m] = {mag[m] * mag[m], 0.0f};
        acfFft_.inverse(aspec_.data(), acf_);
        const double r0 = acf_[0];
        if (!(r0 > 1e-20)) return 0.0;
        // circular ACF: keep lags <= N/3 where the wrapped term is < 6%
        const int lo = std::max(2, static_cast<int>(sampleRate_ / f0MaxHz_));
        const int hi = std::min(nFull_ / 3, static_cast<int>(sampleRate_ / f0MinHz_) + 1);
        if (hi <= lo + 2) return 0.0;
        // periodic-Hann window autocorrelation (analytic, normalized)
        auto rw = [&](double t) {
            const double x = t / nFull_;
            return (1.0 - x) * (2.0 / 3.0 + std::cos(2.0 * M_PI * x) / 3.0) + std::sin(2.0 * M_PI * x) / (2.0 * M_PI);
        };
        auto nr = [&](int t) { return acf_[t] / (r0 * rw(t)); };
        double best = 0.0;
        for (int t = lo; t <= hi; ++t) {
            const double v = nr(t);
            if (v > best && v >= nr(t - 1) && v >= nr(t + 1)) best = v;
        }
        lastPeriodicity_ = best;
        if (best < 0.45) return 0.0;  // Praat's voicing threshold
        for (int t = lo; t <= hi; ++t) {
            const double v = nr(t), a = nr(t - 1), b = nr(t + 1);
            if (v >= 0.85 * best && v >= a && v >= b) {
                const double den = a - 2.0 * v + b;
                const double d = den < 0.0 ? 0.5 * (a - b) / den : 0.0;
                const double period = t + std::clamp(d, -.5, .5);
                if (persistentShort && sampleRate_ / period > 500) {
                    const double farther = 8 * period;
                    const int integer = int(std::floor(farther));
                    if (integer + 1 >= nFull_ / 3) continue;
                    const double fraction = farther - integer;
                    if (nr(integer) * (1 - fraction) + nr(integer + 1) * fraction < .75) continue;
                }
                return t + std::clamp(d, -0.5, 0.5);
            }
        }
        return 0.0;
    }

    int nFull_, binsFull_, sampleRate_;
    double lastPeriodicity_ = 0, f0MinHz_ = 60, f0MaxHz_ = 1100;
    RealFFT acfFft_;
    std::vector<std::complex<float>> aspec_;
    float* acf_ = nullptr;
};
}  // namespace pbshift
