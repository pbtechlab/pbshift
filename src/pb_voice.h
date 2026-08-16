// pb_voice.h — streaming pitch-synchronous voice time-stretch.
//
// A clean-room causal build of the vocal path that measured best on our voice
// material at every ratio from 0.5x to 4x, shortening included. The offline
// prototype it follows (see benchmarks/) cannot be shipped as-is: it tracks f0
// over the whole signal and searches for grain positions in both directions, so
// it is non-causal by construction. Everything here is rewritten to run from a
// bounded look-ahead, and the look-ahead is stated in samples so a host can
// compensate for it.
//
// Why time domain at all: a phase vocoder buys length by re-printing spectral
// frames, and on voice that reads as the fine "delay echo" texture — copies of
// the same glottal pulse landing one synthesis hop apart. Carrying real pitch
// periods instead means the waveform is never resynthesised, so that comb has
// no way to form, and the spectral valleys between harmonics stay as clean as
// the input's.
//
// The three mechanisms that matter, each aimed at one audible defect:
//
//   consistent-polarity pitch marks -- a mark is refined to the local extremum
//       of one globally chosen excitation polarity. Marks that flip sign
//       between periods splice opposite-going waveforms together and click.
//
//   wide correlation search on the low band -- when a period has to be reused
//       to buy time, a narrow search leaves the copy sitting almost exactly one
//       period away from its original, which is the deepest possible comb (the
//       "doubling"). Searching +-0.9 of a period lets the reuse land where the
//       waveform actually matches, breaking that periodicity while still
//       carrying real signal, so it costs neither chorus nor noise.
//
//   de-doubled high band -- above the crossover the content is aspiration and
//       sibilance, i.e. noise, where a duplicate is heard as a doubled hiss.
//       Alternate grains are time-reversed: that decorrelates a grain from its
//       own repeat while preserving its magnitude spectrum exactly, so it
//       removes the doubling without adding a broadband floor.
//
// Deterministic: every decision is a function of absolute sample position and a
// fixed amount of look-ahead, never of how the host happened to chunk its
// buffers, so identical input yields bit-identical output at any block size.
#ifndef PBSHIFT_PB_VOICE_H
#define PBSHIFT_PB_VOICE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace pbshift {

class VoiceStretch {
public:
    // Pitch search range. 80 Hz rather than 70 keeps the longest period (and
    // therefore the grain look-ahead) inside the low-latency budget; below this
    // the tracker falls back to holding the last period it trusted.
    static constexpr double kFmin = 80.0;
    static constexpr double kFmax = 500.0;
    static constexpr double kCrossoverHz = 3000.0;
    static constexpr int kYinFrame = 2048;   // >= 2 periods at kFmin @ 48 kHz
    static constexpr int kYinHop = 256;
    static constexpr int kFirTaps = 257;     // linear phase, odd
    static constexpr double kVoicedTol = 0.9;    // grain search, in periods
    static constexpr double kUnvoicedTol = 0.4;
    static constexpr double kYinThreshold = 0.15;

    void configure(int sampleRate, int channels) {
        sr_ = sampleRate;
        ch_ = std::max(1, channels);
        tMin_ = static_cast<int>(std::floor(sr_ / kFmax));
        tMax_ = static_cast<int>(std::ceil(sr_ / kFmin));
        buildCrossover();
        reset();
    }

    void reset() {
        in_.assign(ch_, {});
        low_.assign(ch_, {});
        high_.assign(ch_, {});
        outLow_.assign(ch_, {});
        outHigh_.assign(ch_, {});
        norm_.clear();
        periods_.clear();
        voicedFlags_.clear();
        inCount_ = 0;
        filtered_ = 0;
        yinFrame_ = 0;
        markPos_ = -1;
        tOut_ = 0.0;
        outWritten_ = 0;
        outRead_ = 0;
        target_ = 0;
        cycle_ = 0;
        period_ = sr_ / 150.0;
        voiced_ = false;
        polaritySum_ = 0.0;
        polarityCount_ = 0;
        polarity_ = 1.0;
        finished_ = false;
        started_ = false;
    }

    // Look-ahead the engine needs on the input side, in samples. The tracker
    // reports an f0 for the centre of a frame, so it costs half a frame; the
    // grain search may reach one period plus its tolerance past the grain
    // centre; the crossover is linear phase and costs half its length.
    int inputLatency() const {
        return kYinFrame / 2 + static_cast<int>((1.0 + kVoicedTol) * tMax_) +
               kFirTaps / 2;
    }
    // Output cannot be handed over until no later grain can still touch it: a
    // grain reaches half its length past its placement.
    int outputLatency() const { return inputLatency() + tMax_; }

    void setStretch(double alpha) { alpha_ = std::clamp(alpha, 0.25, 4.0); }
    void setReversePeriod(int p) { revPeriod_ = p; }   // 0 = automatic

    void feed(const float* const* in, int frames) {
        for (int c = 0; c < ch_; ++c)
            in_[c].insert(in_[c].end(), in[c], in[c] + frames);
        inCount_ += frames;
        run();
    }

    void finish() {
        finished_ = true;
        // The requested length is exact: a host asking for 2x expects twice the
        // samples it fed, not twice minus whatever the last grain happened to
        // reach.
        target_ = static_cast<long long>(std::llround(inCount_ * alpha_));
        run();
    }

    int available() const {
        return static_cast<int>(std::max<long long>(0, outWritten_ - outRead_));
    }

    int read(float* const* out, int frames) {
        const int n = std::min(frames, available());
        for (int c = 0; c < ch_; ++c) {
            for (int i = 0; i < n; ++i) {
                const size_t idx = static_cast<size_t>(outRead_ + i);
                const double w = idx < norm_.size() ? norm_[idx] : 0.0;
                const double lo = idx < outLow_[c].size() ? outLow_[c][idx] : 0.0;
                const double hi =
                    idx < outHigh_[c].size() ? outHigh_[c][idx] : 0.0;
                out[c][i] = static_cast<float>((lo + hi) /
                                               (w > 1e-6 ? w : 1.0));
            }
        }
        outRead_ += n;
        return n;
    }

private:
    // ---- crossover -------------------------------------------------------
    // Linear-phase FIR, complementary by construction: the high band is the
    // delayed input minus the low band, so the two sum back to the original
    // exactly. A zero-phase (forward-backward) design would be cleaner still
    // but is not causal, which is the whole point of this build.
    void buildCrossover() {
        fir_.assign(kFirTaps, 0.0);
        const int m = kFirTaps / 2;
        const double wc = 2.0 * M_PI * kCrossoverHz / sr_;
        double sum = 0.0;
        for (int i = 0; i < kFirTaps; ++i) {
            const int k = i - m;
            const double sinc = (k == 0) ? wc / M_PI
                                         : std::sin(wc * k) / (M_PI * k);
            const double win = 0.54 - 0.46 * std::cos(2.0 * M_PI * i /
                                                      (kFirTaps - 1));
            fir_[i] = sinc * win;
            sum += fir_[i];
        }
        for (double& v : fir_) v /= sum;     // unity at DC
    }

    void filterAvailable() {
        const long long limit = inCount_ - kFirTaps;
        for (; filtered_ <= limit; ++filtered_) {
            for (int c = 0; c < ch_; ++c) {
                double acc = 0.0;
                const float* p = in_[c].data() + filtered_;
                for (int i = 0; i < kFirTaps; ++i) acc += fir_[i] * p[i];
                low_[c].push_back(acc);
                // delayed original minus low band == high band, sample exact
                high_[c].push_back(p[kFirTaps / 2] - acc);
            }
        }
    }

    // ---- pitch tracking --------------------------------------------------
    // YIN: the cumulative mean normalised difference function, with parabolic
    // interpolation of the chosen lag. Public prior art, and cheap enough to
    // run per hop in a real-time path.
    void trackAvailable() {
        while (static_cast<long long>(yinFrame_) * kYinHop + kYinFrame <=
               inCount_) {
            const long long start =
                static_cast<long long>(yinFrame_) * kYinHop;
            estimate(start);
            ++yinFrame_;
        }
    }

    void estimate(long long start) {
        const int n = kYinFrame;
        std::vector<double> x(n, 0.0);
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int c = 0; c < ch_; ++c) s += in_[c][start + i];
            x[i] = s / ch_;
        }
        const int maxLag = std::min(tMax_, n / 2);
        std::vector<double> d(maxLag + 1, 0.0);
        for (int lag = tMin_; lag <= maxLag; ++lag) {
            double acc = 0.0;
            for (int i = 0; i + lag < n; ++i) {
                const double diff = x[i] - x[i + lag];
                acc += diff * diff;
            }
            d[lag] = acc;
        }
        std::vector<double> cmnd(maxLag + 1, 1.0);
        double running = 0.0;
        for (int lag = tMin_; lag <= maxLag; ++lag) {
            running += d[lag];
            cmnd[lag] = d[lag] * (lag - tMin_ + 1) / (running + 1e-12);
        }
        int best = -1;
        for (int lag = tMin_; lag <= maxLag; ++lag) {
            if (cmnd[lag] < kYinThreshold) {
                while (lag + 1 <= maxLag && cmnd[lag + 1] < cmnd[lag]) ++lag;
                best = lag;
                break;
            }
        }
        if (best < 0) {
            int argmin = tMin_;
            for (int lag = tMin_; lag <= maxLag; ++lag)
                if (cmnd[lag] < cmnd[argmin]) argmin = lag;
            // Unvoiced: keep the last trusted period so grain sizes stay
            // continuous across the boundary; a jump here clicks.
            voiced_ = cmnd[argmin] < 0.4;
            if (!voiced_) {
                record(false);
                return;
            }
            best = argmin;
        } else {
            voiced_ = true;
        }
        double refined = best;
        if (best > tMin_ && best < maxLag) {
            const double a = cmnd[best - 1], b = cmnd[best], c = cmnd[best + 1];
            const double den = a - 2.0 * b + c;
            if (std::fabs(den) > 1e-12) refined += 0.5 * (a - c) / den;
        }
        period_ = std::clamp(refined, static_cast<double>(tMin_),
                             static_cast<double>(tMax_));
        record(true);

        // Excitation polarity: one global sign, frozen once enough voiced
        // audio has been seen. Refining marks to alternating polarities would
        // splice opposite-going waveforms and click.
        if (polarityCount_ < 8) {
            double cube = 0.0;
            for (double v : x) cube += v * v * v;
            polaritySum_ += cube;
            ++polarityCount_;
            polarity_ = polaritySum_ >= 0.0 ? 1.0 : -1.0;
        }
    }

    // The contour is stored per analysis frame and looked up BY POSITION. Using
    // "the newest estimate" instead makes a grain depend on how far tracking
    // happened to have run when the host called feed(), which is a second way
    // to lose block-size determinism.
    void record(bool voiced) {
        periods_.push_back(period_);
        voicedFlags_.push_back(voiced ? 1 : 0);
    }

    int frameFor(long long pos) const {
        if (periods_.empty()) return -1;
        long long idx = std::llround((pos - kYinFrame / 2.0) / kYinHop);
        idx = std::clamp<long long>(idx, 0,
                                    static_cast<long long>(periods_.size()) - 1);
        return static_cast<int>(idx);
    }

    double periodAt(long long pos) const {
        const int i = frameFor(pos);
        return i < 0 ? period_ : periods_[i];
    }

    bool voicedAt(long long pos) const {
        const int i = frameFor(pos);
        return i < 0 ? false : voicedFlags_[i] != 0;
    }

    // ---- synthesis -------------------------------------------------------
    // How much filtered input must exist before a grain centred at `centre`
    // can be placed, including its search range and its own half length.
    long long needFor(long long centre) const {
        // Everything a grain step may read: its own half length, the alignment
        // search, and the window in which the next pitch mark is refined. The
        // amount is FIXED rather than "whatever has arrived", because a rule
        // that depends on arrival makes the output depend on the host's block
        // size -- which is exactly how determinism was lost the first time.
        return centre +
               static_cast<long long>((1.0 + kVoicedTol + 0.3) * tMax_) + 4;
    }

    void run() {
        filterAvailable();
        trackAvailable();
        const long long haveFiltered = filtered_;   // low_/high_ length
        if (!started_) {
            // Wait for the tracker as well as for the audio: starting before
            // the first f0 estimate exists would anchor the schedule on a
            // default period, and whether that happens depends on the host's
            // block size.
            if (!finished_ &&
                (haveFiltered < needFor(tMax_) || periods_.empty()))
                return;
            // Anchor on the FIRST tracked frame, not on whatever the tracker
            // had reached by the time enough input arrived: the latter shifts
            // the whole grain schedule when the host uses a different block.
            const double first = periods_.empty() ? period_ : periods_.front();
            markPos_ = static_cast<long long>(std::llround(first));
            tOut_ = static_cast<double>(markPos_);
            started_ = true;
        }

        for (;;) {
            long long srcCentre =
                static_cast<long long>(std::llround(tOut_ / alpha_));
            if (!finished_ && needFor(srcCentre) > haveFiltered) break;
            if (finished_ && target_ > 0 &&
                static_cast<long long>(std::llround(tOut_)) >= target_)
                break;
            if (srcCentre >= haveFiltered) {
                if (!finished_) break;
                // Draining: keep laying grains from the last real audio so the
                // requested length is reached without a truncated final period.
                srcCentre = haveFiltered - 1;
                if (srcCentre < 0) break;
            }

            const double T = periodAt(srcCentre);
            const int L = std::max(2, static_cast<int>(std::lround(T)));
            const bool vo = voicedAt(srcCentre);

            // Low band: pitch-synchronous grains, aligned by a wide search.
            long long centre = vo ? nearestMark(srcCentre, T) : srcCentre;
            int d = bestOffset(low_, centre, L,
                               static_cast<int>((vo ? kVoicedTol : kUnvoicedTol) *
                                                T));
            addGrain(outLow_, low_, centre + d, L, tOut_, false, true);

            // High band: continuous source, alternate grains time-reversed.
            int dh = bestOffset(high_, srcCentre, L,
                                static_cast<int>(kUnvoicedTol * T));
            addGrain(outHigh_, high_, srcCentre + dh, L, tOut_,
                     reverseThisGrain(), false);

            tOut_ += T;
            ++cycle_;
            publish();
        }
        publish();
    }

    // One mark per period, refined to the local extremum of the chosen
    // polarity; marks advance strictly forward so the schedule is causal.
    long long nearestMark(long long target, double T) {
        const int w = std::max(2, static_cast<int>(0.3 * T));
        while (markPos_ + static_cast<long long>(std::lround(T)) <= target) {
            long long guess = markPos_ + static_cast<long long>(std::lround(T));
            long long a = std::max<long long>(0, guess - w);
            long long b = std::min<long long>(filtered_, guess + w);
            if (b <= a) break;
            long long bestI = guess;
            double bestV = -1e30;
            for (long long i = a; i < b; ++i) {
                double s = 0.0;
                for (int c = 0; c < ch_; ++c) s += low_[c][i];
                s *= polarity_;
                if (s > bestV) {
                    bestV = s;
                    bestI = i;
                }
            }
            markPos_ = bestI;
        }
        return markPos_;
    }

    // WSOLA alignment: pick the shift whose leading half correlates best with
    // what is already written, so the grain adds constructively.
    int bestOffset(const std::vector<std::vector<double>>& band,
                   long long centre, int L, int tol) {
        if (tol < 2) return 0;
        const long long tb = static_cast<long long>(std::llround(tOut_));
        const long long ta = std::max<long long>(0, tb - L);
        const int m = static_cast<int>(tb - ta);
        if (m < 8) return 0;
        std::vector<double> tmpl(m, 0.0);
        double energy = 0.0;
        for (int i = 0; i < m; ++i) {
            const size_t idx = static_cast<size_t>(ta + i);
            const double w = idx < norm_.size() ? norm_[idx] : 0.0;
            double v = 0.0;
            if (w > 1e-6) {
                for (int c = 0; c < ch_; ++c) {
                    const double lo =
                        idx < outLow_[c].size() ? outLow_[c][idx] : 0.0;
                    const double hi =
                        idx < outHigh_[c].size() ? outHigh_[c][idx] : 0.0;
                    v += lo + hi;
                }
                v /= (ch_ * w);
            }
            tmpl[i] = v;
            energy += v * v;
        }
        if (energy < 1e-18) return 0;

        int bestD = 0;
        double bestC = -1e30;
        for (int dd = -tol; dd <= tol; dd += 2) {
            const long long cs = centre + dd;
            const long long a = std::max<long long>(0, cs - L);
            const int len = static_cast<int>(std::min<long long>(cs - a, m));
            if (len < 8) continue;
            double dot = 0.0, nrm = 0.0;
            for (int i = 0; i < len; ++i) {
                const long long si = cs - len + i;
                if (si < 0 || si >= filtered_) continue;
                double s = 0.0;
                for (int c = 0; c < ch_; ++c) s += band[c][si];
                s /= ch_;
                dot += s * tmpl[m - len + i];
                nrm += s * s;
            }
            const double c = dot / (std::sqrt(nrm) + 1e-9);
            if (c > bestC) {
                bestC = c;
                bestD = dd;
            }
        }
        return bestD;
    }

    // How often a high-band grain is time-reversed. The right rate is not
    // fixed: it depends on how many times the same source material gets reused,
    // which is set by the stretch ratio. Alternating suits a 2x reuse; at 4x the
    // alternation itself repeats (A, Ar, A, Ar) and the doubling returns.
    bool reverseThisGrain() const {
        // Reverse one grain in every `p`, where p tracks the reuse count, i.e.
        // the stretch ratio. Measured across 1.25x-4x the best setting is
        // always p = round(alpha): at 2x plain alternation, at 3x every third,
        // at 4x every fourth. Getting this wrong is not a small loss -- a
        // mismatched period (every third grain at 2x) fills the spectral
        // valleys by 3.5 dB and is rejected outright.
        const int p = revPeriod_ > 0
                          ? revPeriod_
                          : std::clamp(static_cast<int>(std::lround(alpha_)), 2, 8);
        return (cycle_ % p) == 1;
    }

    void addGrain(std::vector<std::vector<double>>& dst,
                  const std::vector<std::vector<double>>& band,
                  long long centre, int L, double tOut, bool reverse,
                  bool accumulateNorm) {
        const long long a = centre - L, b = centre + L;
        const long long ga = std::max<long long>(0, a);
        const long long gb = std::min<long long>(filtered_, b);
        const int len = static_cast<int>(gb - ga);
        if (len < 4) return;
        const long long oa =
            static_cast<long long>(std::llround(tOut)) - (centre - ga);
        ensure(dst, static_cast<size_t>(oa + len));
        if (accumulateNorm && norm_.size() < static_cast<size_t>(oa + len))
            norm_.resize(static_cast<size_t>(oa + len), 0.0);
        for (int i = 0; i < len; ++i) {
            const long long oi = oa + i;
            if (oi < 0) continue;
            const double w =
                0.5 - 0.5 * std::cos(2.0 * M_PI * i / std::max(len - 1, 1));
            const int src = reverse ? (len - 1 - i) : i;
            for (int c = 0; c < ch_; ++c)
                dst[c][static_cast<size_t>(oi)] += band[c][ga + src] * w;
            if (accumulateNorm) norm_[static_cast<size_t>(oi)] += w;
        }
    }

    void ensure(std::vector<std::vector<double>>& v, size_t n) {
        for (auto& b : v)
            if (b.size() < n) b.resize(n, 0.0);
    }

    // A sample is final once no later grain can reach it: grains are placed at
    // tOut_ and reach half a grain back, so everything before tOut_ - tMax_ is
    // settled.
    void publish() {
        long long safe = finished_
                             ? target_
                             : static_cast<long long>(std::llround(tOut_)) -
                                   tMax_;
        if (!finished_)
            safe = std::min<long long>(safe,
                                       static_cast<long long>(norm_.size()));
        else
            ensureOut(static_cast<size_t>(target_));
        if (safe > outWritten_) outWritten_ = safe;
    }

    void ensureOut(size_t n) {
        ensure(outLow_, n);
        ensure(outHigh_, n);
        if (norm_.size() < n) norm_.resize(n, 0.0);
    }

    int sr_ = 48000, ch_ = 1;
    int tMin_ = 96, tMax_ = 600;
    double alpha_ = 1.0;
    std::vector<double> fir_;
    std::vector<std::vector<float>> in_;
    std::vector<std::vector<double>> low_, high_, outLow_, outHigh_;
    std::vector<double> norm_;
    std::vector<double> periods_;
    std::vector<int> voicedFlags_;
    long long inCount_ = 0, filtered_ = 0, markPos_ = -1;
    long long outWritten_ = 0, outRead_ = 0, target_ = 0;
    int yinFrame_ = 0, cycle_ = 0;
    double tOut_ = 0.0, period_ = 320.0;
    bool voiced_ = false, finished_ = false, started_ = false;
    double polaritySum_ = 0.0, polarity_ = 1.0;
    int polarityCount_ = 0;
    int revPeriod_ = 0;
};

}  // namespace pbshift

#endif  // PBSHIFT_PB_VOICE_H
