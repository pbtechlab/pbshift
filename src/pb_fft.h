// Real FFT wrapper around pffft (BSD-like license).
#pragma once
#include <cassert>
#include <complex>
#include <cstring>
#include <vector>

#include "pffft.h"

namespace pbshift {

// Power-of-two real FFT, N >= 32. Forward fills N/2+1 complex bins,
// inverse consumes them. Inverse is unnormalized by pffft; we fold the
// 1/N normalization into inverse() so forward+inverse == identity.
class RealFFT {
public:
    explicit RealFFT(int n) : n_(n) {
        assert(n >= 32 && (n & (n - 1)) == 0);
        setup_ = pffft_new_setup(n, PFFFT_REAL);
        work_ = alloc(n);
        packed_ = alloc(n);
    }
    ~RealFFT() {
        pffft_destroy_setup(setup_);
        pffft_aligned_free(work_);
        pffft_aligned_free(packed_);
    }
    RealFFT(const RealFFT&) = delete;
    RealFFT& operator=(const RealFFT&) = delete;

    int size() const { return n_; }
    int bins() const { return n_ / 2 + 1; }

    // time[n] (aligned not required) -> spec[n/2+1]
    void forward(const float* time, std::complex<float>* spec) {
        pffft_transform_ordered(setup_, time, packed_, work_, PFFFT_FORWARD);
        // pffft ordered real layout: [Re(0), Re(N/2), Re(1), Im(1), ...]
        spec[0] = {packed_[0], 0.0f};
        spec[n_ / 2] = {packed_[1], 0.0f};
        for (int k = 1; k < n_ / 2; ++k)
            spec[k] = {packed_[2 * k], packed_[2 * k + 1]};
    }

    // spec[n/2+1] -> time[n], includes 1/N scaling
    void inverse(const std::complex<float>* spec, float* time) {
        packed_[0] = spec[0].real();
        packed_[1] = spec[n_ / 2].real();
        for (int k = 1; k < n_ / 2; ++k) {
            packed_[2 * k] = spec[k].real();
            packed_[2 * k + 1] = spec[k].imag();
        }
        pffft_transform_ordered(setup_, packed_, time, work_, PFFFT_BACKWARD);
        const float s = 1.0f / static_cast<float>(n_);
        for (int i = 0; i < n_; ++i) time[i] *= s;
    }

    static float* alloc(int n) {
        return static_cast<float*>(pffft_aligned_malloc(sizeof(float) * n));
    }

private:
    int n_;
    PFFFT_Setup* setup_;
    float* work_;
    float* packed_;
};

}  // namespace pbshift
