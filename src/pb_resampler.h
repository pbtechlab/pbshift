// Streaming variable-ratio windowed-sinc resampler (Kaiser window).
// ratio = output_rate / input_rate in the time sense: producing
// out samples at input positions j/ratio. Pitch scale factor = 1/ratio.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace pbshift {

class SincResampler {
public:
    // taps: one-sided tap count (total window = 2*taps)
    explicit SincResampler(int channels, int taps = 32, double beta = 10.0)
        : ch_(channels), taps_(taps), beta_(beta) {
        hist_.assign(ch_, std::vector<float>(2 * taps_, 0.0f));
        histLen_ = 0;
        pos_ = 0.0;
        ib_ = 1.0 / bessel0(beta_);
        // tabulated Kaiser window (the per-tap bessel0 was the RT hot spot)
        kaiserTab_.resize(kTabN + 2);
        for (int i = 0; i <= kTabN; ++i) {
            const double x = static_cast<double>(i) / kTabN;  // 0..1
            kaiserTab_[i] = bessel0(beta_ * std::sqrt(1.0 - x * x)) * ib_;
        }
        kaiserTab_[kTabN + 1] = 0.0;
        // tabulated sine over one period for the non-memoized reference path
        // (irrational ratios never repeat a phase, so std::sin dominates).
        // 16384 samples + linear interp: error < 3e-8, sub-float and far
        // below the interpolation floor.
        sinTab_.resize(kSinN + 2);
        for (int i = 0; i <= kSinN; ++i)
            sinTab_[i] = std::sin(2.0 * M_PI * i / kSinN);
        sinTab_[kSinN + 1] = sinTab_[1];
    }

    void reset() {
        for (auto& h : hist_) std::fill(h.begin(), h.end(), 0.0f);
        histLen_ = 0;
        pos_ = 0.0;
        outCount_ = 0;
        baseIn_ = 0;
        buf_.assign(ch_, {});
    }

    // Push input; internally buffered. in: [channel][frames].
    void feed(const float* const* in, int frames) {
        if (buf_.size() != static_cast<size_t>(ch_)) buf_.resize(ch_);
        for (int c = 0; c < ch_; ++c)
            buf_[c].insert(buf_[c].end(), in[c], in[c] + frames);
    }

    void finish() {
        // pad tail so the last real samples can be interpolated
        if (buf_.size() != static_cast<size_t>(ch_)) buf_.resize(ch_);
        for (int c = 0; c < ch_; ++c)
            buf_[c].insert(buf_[c].end(), 2 * taps_, 0.0f);
    }

    // Produce up to maxOut frames at the given ratio; returns count.
    // out: [channel][frames].
    // Deterministic and chunk-independent: the source position of output j
    // is computed from absolute integer counters (outCount_/ratio), never
    // from an accumulated float, so call-boundary placement cannot change
    // a single output bit. Ratio must stay constant within a stream.
    int process(float* const* out, int maxOut, double ratio) {
        if (buf_.empty() || buf_[0].empty()) return 0;
        const double cutoff = 0.92 * std::min(1.0, ratio);
        if (!(ratio == memoRatio_)) {  // ratio change invalidates the memo
            memoRatio_ = ratio;
            memoIdx_.clear();
            memoSinc_.clear();
            memoW_.clear();
            memoSkip_.clear();
        }
        const long long avail =
            baseIn_ + static_cast<long long>(buf_[0].size());
        int made = 0;
        while (made < maxOut) {
            const double srcAbs = static_cast<double>(outCount_) / ratio;
            const long long i0 = static_cast<long long>(std::floor(srcAbs));
            if (i0 + taps_ >= avail) break;  // need more input
            // srcAbs - i0 is exact (both representable, result in [0,1)),
            // so the fractional source phase keys the tap-weight memo
            const int e = memoEntry(srcAbs - static_cast<double>(i0), cutoff);
            if (e >= 0) {
                // memo replay: same values, same accumulation order as the
                // reference loop below => bit-identical output
                const double* sv = memoSinc_[e].data();
                const double* wv = memoW_[e].data();
                const uint8_t* sk = memoSkip_[e].data();
                const long long iFirst = i0 - taps_ + 1;
                const int jStart = static_cast<int>(
                    std::max<long long>(0, baseIn_ - iFirst));
                for (int c = 0; c < ch_; ++c) {
                    double acc = 0.0;
                    const std::vector<float>& src = buf_[c];
                    size_t idx = static_cast<size_t>(iFirst + jStart - baseIn_);
                    for (int j = jStart; j < 2 * taps_; ++j, ++idx) {
                        if (idx >= src.size()) break;
                        if (!sk[j]) acc += src[idx] * sv[j] * wv[j];
                    }
                    out[c][made] = static_cast<float>(acc);
                }
            } else {
                // reference path (non-recurring phase, memo full)
                for (int c = 0; c < ch_; ++c) {
                    double acc = 0.0;
                    const std::vector<float>& src = buf_[c];
                    for (long long i = i0 - taps_ + 1; i <= i0 + taps_; ++i) {
                        if (i < baseIn_) continue;
                        const size_t idx = static_cast<size_t>(i - baseIn_);
                        if (idx >= src.size()) break;
                        const double d = srcAbs - static_cast<double>(i);
                        const double x = std::abs(d) / taps_;
                        if (x >= 1.0) continue;
                        const double sinc =
                            std::abs(d) < 1e-12
                                ? cutoff
                                : fastSin(M_PI * cutoff * d) / (M_PI * d);
                        const double tx = x * kTabN;
                        const int ti = static_cast<int>(tx);
                        const double tf = tx - ti;
                        const double w =
                            kaiserTab_[ti] * (1.0 - tf) + kaiserTab_[ti + 1] * tf;
                        acc += src[idx] * sinc * w;
                    }
                    out[c][made] = static_cast<float>(acc);
                }
            }
            ++outCount_;
            ++made;
        }
        // drop consumed history (keep taps_ of lookbehind)
        const long long nextI0 = static_cast<long long>(
            std::floor(static_cast<double>(outCount_) / ratio));
        const long long keepFrom = nextI0 - taps_;
        if (keepFrom > baseIn_) {
            const long long drop = keepFrom - baseIn_;
            for (int c = 0; c < ch_; ++c)
                buf_[c].erase(buf_[c].begin(), buf_[c].begin() + drop);
            baseIn_ = keepFrom;
        }
        return made;
    }

    int pending() const {
        return buf_.empty() ? 0 : static_cast<int>(buf_[0].size());
    }

private:
    // ---- per-phase tap-weight memo (bit-identical fast path) ----------
    // The sinc and Kaiser weights depend only on the fractional source
    // phase frac = srcAbs - i0 (and the fixed ratio). For dyadic ratios
    // (pitch +-12/+-24: ratio 2, 0.5, 4, 0.25) frac cycles through a tiny
    // set of exact doubles, so the per-tap std::sin vanishes after the
    // first occurrence of each phase. Weights are computed with exactly
    // the reference expressions: d = frac + (taps-1-j) rounds identically
    // to srcAbs - i because frac is the exact fractional part, so replay
    // is bit-identical. Non-recurring phases (general ratios) fall back to
    // the reference loop once the memo is full; the memo is bounded.
    int memoEntry(double frac, double cutoff) {
        uint64_t key;
        std::memcpy(&key, &frac, sizeof key);
        const auto it = memoIdx_.find(key);
        if (it != memoIdx_.end()) return it->second;
        if (memoIdx_.size() >= kMemoCap) return -1;
        const int e = static_cast<int>(memoSinc_.size());
        memoSinc_.emplace_back(2 * taps_, 0.0);
        memoW_.emplace_back(2 * taps_, 0.0);
        memoSkip_.emplace_back(2 * taps_, uint8_t(0));
        std::vector<double>& sv = memoSinc_.back();
        std::vector<double>& wv = memoW_.back();
        std::vector<uint8_t>& sk = memoSkip_.back();
        for (int j = 0; j < 2 * taps_; ++j) {
            // tap j covers input index i = i0 - taps_ + 1 + j
            const double d = frac + static_cast<double>(taps_ - 1 - j);
            const double x = std::abs(d) / taps_;
            if (x >= 1.0) {
                sk[j] = 1;  // reference loop skips this tap entirely
                continue;
            }
            sv[j] = std::abs(d) < 1e-12
                        ? cutoff
                        : std::sin(M_PI * cutoff * d) / (M_PI * d);
            const double tx = x * kTabN;
            const int ti = static_cast<int>(tx);
            const double tf = tx - ti;
            wv[j] = kaiserTab_[ti] * (1.0 - tf) + kaiserTab_[ti + 1] * tf;
        }
        memoIdx_.emplace(key, e);
        return e;
    }

    static double bessel0(double x) {
        double s = 1.0, t = 1.0;
        for (int k = 1; k < 32; ++k) {
            t *= (x / (2.0 * k)) * (x / (2.0 * k));
            s += t;
            if (t < 1e-18 * s) break;
        }
        return s;
    }

    // Table sine (one period, linear interp). Deterministic pure function
    // of its argument, so chunk-independence / re-render identity hold.
    double fastSin(double theta) const {
        double u = theta * (1.0 / (2.0 * M_PI));
        u -= std::floor(u);  // [0,1)
        const double t = u * kSinN;
        const int i = static_cast<int>(t);
        const double f = t - i;
        return sinTab_[i] * (1.0 - f) + sinTab_[i + 1] * f;
    }

    static constexpr int kTabN = 4096;
    static constexpr int kSinN = 16384;
    static constexpr size_t kMemoCap = 64;
    std::vector<double> sinTab_;
    double memoRatio_ = -1.0;  // impossible ratio: first process() resets
    std::unordered_map<uint64_t, int> memoIdx_;
    std::vector<std::vector<double>> memoSinc_, memoW_;
    std::vector<std::vector<uint8_t>> memoSkip_;
    std::vector<double> kaiserTab_;
    int ch_, taps_;
    double beta_, ib_;
    std::vector<std::vector<float>> hist_;
    int histLen_;
    double pos_;
    long long outCount_ = 0;  // absolute output frames produced
    long long baseIn_ = 0;    // absolute input index of buf_[0]
    std::vector<std::vector<float>> buf_;
};

}  // namespace pbshift
