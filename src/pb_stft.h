// Shared analysis front-end: one windowed frame -> spectrum + phase gradients
// via reassignment operators (3 real FFTs: w, dw, tw).
//
// Convention: the windowed frame is circularly rotated so the window center
// sits at index 0 before the FFT ("center-referenced" phases). This makes
// phase gradients well-conditioned and hop-independent.
#pragma once
#include <complex>
#include <vector>

#include "pb_fft.h"
#include "pb_window.h"

namespace pbshift {

struct AnalysisFrame {
    std::vector<std::complex<float>> X;  // bins() spectrum (center-referenced)
    std::vector<float> mag;              // |X|
    std::vector<float> omega;            // instantaneous freq, rad/input-sample
    std::vector<float> tau;              // local group delay, samples from window center

    void resize(int bins) {
        X.resize(bins);
        mag.resize(bins);
        omega.resize(bins);
        tau.resize(bins);
    }
};

class AnalysisFrontend {
public:
    AnalysisFrontend(int fftSize, const WindowSet& win)
        : n_(fftSize), fft_(fftSize), win_(win) {
        buf_ = RealFFT::alloc(n_);
        rot_ = RealFFT::alloc(n_);
        Xd_.resize(fft_.bins());
        Xt_.resize(fft_.bins());
    }
    ~AnalysisFrontend() {
        pffft_aligned_free(buf_);
        pffft_aligned_free(rot_);
    }
    AnalysisFrontend(const AnalysisFrontend&) = delete;
    AnalysisFrontend& operator=(const AnalysisFrontend&) = delete;

    int bins() const { return fft_.bins(); }

    // frame: n_ contiguous input samples (window applied here).
    void analyze(const float* frame, AnalysisFrame& out) {
        out.resize(fft_.bins());
        const int c = n_ / 2;  // window center (periodic Hann)
        // main spectrum
        windowRotate(frame, win_.w.data(), c);
        fft_.forward(rot_, out.X.data());
        // derivative-window spectrum
        windowRotate(frame, win_.dw.data(), c);
        fft_.forward(rot_, Xd_.data());
        // time-weighted-window spectrum
        windowRotate(frame, win_.tw.data(), c);
        fft_.forward(rot_, Xt_.data());

        const int B = fft_.bins();
        const float binW = 2.0f * static_cast<float>(M_PI) / n_;
        for (int m = 0; m < B; ++m) {
            const std::complex<float> X = out.X[m];
            const float e = std::norm(X) + 1e-24f;
            out.mag[m] = std::sqrt(std::norm(X));
            // reassignment operators (center-referenced). By parts:
            // X_dw = j(omega_m - Omega) W  =>  omega_hat = omega_m - Im(X_dw X*)/|X|^2
            // X_tw(delta at d) = d w e^{-j w d} =>  tau_hat = Re(X_tw X*)/|X|^2
            const std::complex<float> pd = Xd_[m] * std::conj(X);
            const std::complex<float> pt = Xt_[m] * std::conj(X);
            out.omega[m] = binW * m - pd.imag() / e;
            out.tau[m] = pt.real() / e;
        }
    }

private:
    // rot_[i] = frame[(i + c) mod n] * win[(i + c) mod n]
    void windowRotate(const float* frame, const float* w, int c) {
        for (int i = 0; i < n_ - c; ++i) rot_[i] = frame[i + c] * w[i + c];
        for (int i = 0; i < c; ++i) rot_[n_ - c + i] = frame[i] * w[i];
    }

    int n_;
    RealFFT fft_;
    const WindowSet& win_;
    float* buf_;
    float* rot_;
    std::vector<std::complex<float>> Xd_, Xt_;
};

// WOLA synthesis accumulator with dual (signal + window-power) accumulation:
// exact amplitude normalization for any hop and at stream edges.
class WolaSynth {
public:
    WolaSynth(int fftSize, const WindowSet& win, int capacity)
        : n_(fftSize), fft_(fftSize), win_(win),
          sig_(capacity, 0.0f), norm_(capacity, 0.0f), cap_(capacity) {
        time_ = RealFFT::alloc(n_);
        unrot_ = RealFFT::alloc(n_);
        // analysis window * synthesis window product profile (both Hann here)
        wprod_.resize(n_);
        for (int i = 0; i < n_; ++i) wprod_[i] = win_.w[i] * win_.w[i];
    }
    ~WolaSynth() {
        pffft_aligned_free(time_);
        pffft_aligned_free(unrot_);
    }
    WolaSynth(const WolaSynth&) = delete;
    WolaSynth& operator=(const WolaSynth&) = delete;

    // Add frame with spectrum Y (center-referenced) at output center position.
    // Ring cells ahead of the high-water mark are zeroed lazily here; read()
    // never clears, so a region may be read, then extended by later frames,
    // and read again (needed at the process/flush boundary).
    void addFrame(const std::complex<float>* Y, long long center) {
        fft_.inverse(Y, time_);
        const int c = n_ / 2;
        // undo center rotation: unrot[(i + c) mod n] = time[i]
        for (int i = 0; i < n_ - c; ++i) unrot_[i + c] = time_[i];
        for (int i = 0; i < c; ++i) unrot_[i] = time_[n_ - c + i];
        const long long start = center - c;
        const long long end = start + n_;
        for (long long p = highWater_; p < end; ++p) {
            const int idx = static_cast<int>(p % cap_);
            sig_[idx] = 0.0f;
            norm_[idx] = 0.0f;
        }
        if (end > highWater_) highWater_ = end;
        for (int i = 0; i < n_; ++i) {
            const long long p = start + i;
            if (p < 0) continue;
            const int idx = static_cast<int>(p % cap_);
            sig_[idx] += unrot_[i] * win_.w[i];
            norm_[idx] += wprod_[i];
        }
    }

    // Read finalized samples [pos, pos+count) without clearing.
    void read(long long pos, int count, float* out) const {
        for (int i = 0; i < count; ++i) {
            const long long p = pos + i;
            if (p >= highWater_) {
                out[i] = 0.0f;
                continue;
            }
            const int idx = static_cast<int>(p % cap_);
            const float nrm = norm_[idx];
            out[i] = nrm > 1e-9f ? sig_[idx] / nrm : 0.0f;
        }
    }

    void clear() {
        std::fill(sig_.begin(), sig_.end(), 0.0f);
        std::fill(norm_.begin(), norm_.end(), 0.0f);
        highWater_ = 0;
    }

private:
    int n_;
    RealFFT fft_;
    const WindowSet& win_;
    std::vector<float> sig_, norm_;
    std::vector<float> wprod_;
    int cap_;
    long long highWater_ = 0;
    float* time_;
    float* unrot_;
};

}  // namespace pbshift
