// Pitch-synchronous speech time stretch with phase-aligned local waveform fits.
// The measured pitch contour is retained until its corresponding output is read.
#ifndef PBSHIFT_SPEECH_TIME_H
#define PBSHIFT_SPEECH_TIME_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <vector>

#include "pb_voice.h"

namespace pbshift {

// Pitch-synchronous waveform fitting with configured audio working rings.
// Retain source pitch metadata until the corresponding output is consumed.
class SpeechTimeProcessor {
   public:
    // Pitch search range. 80 Hz rather than 70 keeps the longest period (and
    // therefore the grain look-ahead) inside the low-latency budget; below this
    // the tracker falls back to holding the last period it trusted.
    static constexpr double kFmin = 80.0;
    static constexpr double kFmax = 500.0;
    static constexpr double kCrossoverHz = 3000.0;
    static constexpr int kYinFrameMin = 2048;
    static constexpr int kYinHop = 256;
    static constexpr int kFirTaps = 257;       // linear phase, odd
    static constexpr double kVoicedTol = 0.9;  // grain search, in periods
    static constexpr double kUnvoicedTol = 0.4;
    static constexpr double kYinThreshold = 0.15;

    void configure(int sampleRate, int channels, int lookaheadBudget = 0) {
        boundedScheduling_ = lookaheadBudget > 0;
        sr_ = sampleRate;
        ch_ = std::max(1, channels);
        tMin_ = static_cast<int>(std::floor(sr_ / kFmax));
        tMax_ = static_cast<int>(std::ceil(sr_ / kFmin));
        // The frame has to hold at least two of the longest period the tracker
        // is allowed to find, or the search silently stops short of kFmin and
        // the pitch floor rises with the sample rate: a fixed 2048 caps the
        // search at 1024 lags, which is only 94 Hz at 96 kHz. Rounded up to a
        // power of two, this is exactly 2048 at 44.1 and 48 kHz, so those rates
        // render bit-identically to before.
        frameLen_ = static_cast<int>(pow2(std::max<long long>(kYinFrameMin, 3LL * tMax_)));
        if (lookaheadBudget > 0) {
            while (tMax_ > tMin_ && requiredInputLookahead() > lookaheadBudget) {
                --tMax_;
                frameLen_ = int(pow2(std::max<long long>(kYinFrameMin, 3LL * tMax_)));
            }
        }
        buildCrossover();

        // Each capacity covers the widest span of positions that is live at
        // once, rounded up to a power of two so indexing is a mask.
        inCap_ = pow2(4LL * frameLen_ + 4LL * tMax_);
        // Generous on purpose. The filter runs as fast as input arrives while
        // synthesis is paced by its own budget, so the gap between them swings
        // by a whole feed call; a window that only just covers the algorithm's
        // reach turns any transient lag into reads of recycled audio.
        bandCap_ = pow2(32LL * tMax_ + kFirTaps + frameLen_);
        // Output has to hold whatever the host has not read yet. A host that
        // reads every block needs only a few grains; this allows several
        // seconds of backlog, and production stalls rather than overwrites if
        // even that is exceeded.
        outCap_ = pow2(std::max<long long>(1LL << 18, 16LL * tMax_));
        ctrCap_ = pow2(8192);

        in_.resize(ch_);
        low_.resize(ch_);
        high_.resize(ch_);
        outLow_.resize(ch_);
        outHigh_.resize(ch_);
        for (int c = 0; c < ch_; ++c) {
            in_[c].init(inCap_);
            low_[c].init(bandCap_);
            high_[c].init(bandCap_);
            outLow_[c].init(outCap_);
            outHigh_[c].init(outCap_);
        }
        lowMono_.init(bandCap_);
        highMono_.init(bandCap_);
        norm_.init(outCap_);
        highNorm_.init(outCap_);
        periods_.init(ctrCap_);
        metadataCapacity_ = ctrCap_;
        metadataOldest_ = 0;
        metadataF0_.init(metadataCapacity_);
        voicedFlags_.init(ctrCap_);

        // Preallocate processing scratch. Retained metadata may grow when the
        // caller postpones output reads; regular draining bounds that history.
        scratchX_.assign(frameLen_, 0.0);
        scratchD_.assign(tMax_ + 2, 0.0);
        scratchC_.assign(tMax_ + 2, 0.0);
        scratchT_.assign(2 * tMax_ + 8, 0.0);
        harmonicMono_.assign(4 * tMax_ + 32, 0);
        gProjectionIndex_.resize(size_t(4 * tMax_ + 32) * 8);
        gProjectionFraction_.resize(size_t(4 * tMax_ + 32) * 8);
        gMatchSource_.resize(2 * tMax_ + 8);
        gMatchReference_.resize(2 * tMax_ + 8);
        gMatchIndex_.resize(2 * tMax_ + 8);
        gMatchBasis_.resize(2 * tMax_ + 8);
        if (boundedScheduling_) {
            sourcePhase_.init(ctrCap_);
            const size_t history = size_t(std::ceil(10.0 * tMax_ / kYinHop)) + 4;
            gSourceFrequency_.resize(history);
            gSourcePhase_.resize(history);
            gSourceVoiced_.resize(history);
        }
        reset();
    }

    static long long pow2(long long n) {
        long long p = 1;
        while (p < n) p <<= 1;
        return p;
    }

    void reset() {
        metadataF0_.clear();
        metadataOldest_ = 0;
        lagFraction_ = lagRemainder_ = 0;
        ratioChanges_.clear();
        sourcePhase_.clear();
        for (int c = 0; c < ch_; ++c) {
            in_[c].clear();
            low_[c].clear();
            high_[c].clear();
            outLow_[c].clear();
            outHigh_[c].clear();
        }
        lowMono_.clear();
        highMono_.clear();
        norm_.clear();
        highNorm_.clear();
        periods_.clear();
        voicedFlags_.clear();
        frames_ = 0;
        inCount_ = 0;
        filtered_ = 0;
        yinFrame_ = 0;
        markPos_ = -1;
        tOut_ = 0.0;
        outAnchor_ = 0.0;
        srcAnchor_ = 0;
        outWritten_ = 0;
        outRead_ = 0;
        target_ = 0;
        written_ = 0;
        curLag_ = 0;
        estimating_ = false;
        drainTracker_ = false;
        stage_ = Stage::Pick;
        searching_ = false;
        gFiltered_ = 0;
        cycle_ = 0;
        period_ = sr_ / 150.0;
        voiced_ = false;
        polarity_ = 1.0;
        finished_ = false;
        started_ = false;
    }

    // Look-ahead the engine needs on the input side, in samples. The tracker
    // reports an f0 for the centre of a frame, so it costs half a frame; the
    // grain search may reach one period plus its tolerance past the grain
    // centre; the crossover is linear phase and costs half its length.
    int requiredInputLookahead() const {
        return std::max(frameLen_ / 2 + kYinHop / 2, int(std::ceil((1 + kVoicedTol + .3) * tMax_)) + 4 + kFirTaps);
    }
    double inputFundamental(long long source) const {
        if (frames_ == 0) return 0;
        const long long frame = std::clamp(idealFrame(source), 0LL, frames_ - 1);
        return frame < metadataOldest_ ? 0 : metadataF0_.at(frame);
    }
    void releaseFundamentalBefore(long long source) {
        metadataOldest_ = std::max(metadataOldest_, std::min(frames_, std::max(0LL, idealFrame(source) - 2)));
    }
    double minimumTrackedHz() const { return double(sr_) / tMax_; }
    double maximumTrackedHz() const { return double(sr_) / tMin_; }
    int inputLatency() const { return frameLen_ / 2 + static_cast<int>((1.0 + kVoicedTol) * tMax_) + kFirTaps / 2; }
    // Output cannot be handed over until no later grain can still touch it: a
    // grain reaches half its length past its placement.
    int outputLatency() const { return inputLatency() + tMax_; }

    // The whole range the engine can hand over, not the range this engine was
    // tuned on. A narrower clamp here is silent and breaks the exact-length
    // guarantee: Voice + 4x + a pitch shift drove alpha to 8, the clamp made it
    // 4, and the post-resampler then halved that -- the host asked for 4x and
    // got 2x, with the whole stream off its timeline rather than just the tail.
    void setPitch(double) {}
    void allowExternalExtrapolation() {}
    void setStretch(double alpha) {
        const double next = std::clamp(alpha, 0.02, 64.0);
        if (next != alpha_ && started_) {
            // Re-anchor so the source pointer CONTINUES from where it is. The
            // mapping is absolute (output position over ratio), so changing the
            // ratio mid-stream would otherwise teleport the read position --
            // halving the ratio after ten seconds jumps it ten seconds ahead of
            // any audio that exists, and the insert goes silent until the input
            // catches up.
            srcAnchor_ = mapSource();
            outAnchor_ = tOut_;
        }
        alpha_ = next;
    }
    void setStretchAtInput(double alpha, long long inputPosition) {
        const double next = std::clamp(alpha, .02, 64.0);
        const double scheduled = ratioChanges_.empty() ? alpha_ : ratioChanges_.back().ratio;
        if (next == scheduled) return;
        if (!started_ && inputPosition == 0) {
            alpha_ = next;
            return;
        }
        RatioChange change{inputPosition, next};
        if (!ratioChanges_.empty() && ratioChanges_.back().at == inputPosition)
            ratioChanges_.back() = change;
        else
            ratioChanges_.push_back(change);
    }
    void setReversePeriod(int p) { revPeriod_ = p; }  // 0 = automatic

    void feed(const float* const* in, int frames) {
        // Consume in chunks no larger than half the input ring. A host is
        // allowed to hand over an entire file in one call, and writing that
        // straight in would overwrite samples the crossover has not read yet.
        const int chunk = static_cast<int>(std::max<long long>(1, inCap_ / 2));
        int done = 0;
        while (done < frames) {
            const int k = std::min(chunk, frames - done);
            for (int c = 0; c < ch_; ++c)
                for (int i = 0; i < k; ++i) in_[c].at(inCount_ + i) = in[c][done + i];
            inCount_ += k;
            done += k;
            run(k);
        }
    }

    void finish(long long requestedTarget = -1) {
        finished_ = true;
        // Draining: no budget applies any more, so let the tracker complete.
        drainTracker_ = true;
        // The requested length is exact: a host asking for 2x expects twice the
        // samples it fed, not twice minus whatever the last grain happened to
        // reach.
        target_ = requestedTarget >= 0 ? requestedTarget : static_cast<long long>(std::llround(inCount_ * alpha_));
        run(frameLen_);
    }

    int available() const { return static_cast<int>(std::max<long long>(0, outWritten_ - outRead_)); }

    int read(float* const* out, int frames) {
        const int n = std::min(frames, available());
        for (int c = 0; c < ch_; ++c) {
            for (int i = 0; i < n; ++i) {
                const long long idx = outRead_ + i;
                const double w = norm_.at(idx);
                const double hw = highNorm_.at(idx);
                out[c][i] = static_cast<float>((!boundedScheduling_ || hw == w)
                                                   ? (outLow_[c].at(idx) + outHigh_[c].at(idx)) / (w > 1e-6 ? w : 1.0)
                                                   : outLow_[c].at(idx) / (w > 1e-6 ? w : 1.0) +
                                                         outHigh_[c].at(idx) / (hw > 1e-6 ? hw : 1.0));
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
            const double sinc = (k == 0) ? wc / M_PI : std::sin(wc * k) / (M_PI * k);
            const double win = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (kFirTaps - 1));
            fir_[i] = sinc * win;
            sum += fir_[i];
        }
        for (double& v : fir_) v /= sum;  // unity at DC
    }

    void filterAvailable() {
        if (boundedScheduling_) {
            const int delay = kFirTaps / 2;
            const long long limit = finished_ ? inCount_ - 1 : inCount_ - delay - 1;
            for (; filtered_ <= limit; ++filtered_) {
                const long long start = filtered_ - delay;
                const bool interior = start >= 0 && start + kFirTaps <= inCount_;
                double monoLow = 0, monoHigh = 0;
                for (int c = 0; c < ch_; ++c) {
                    double low = 0;
                    if (interior)
                        for (int i = 0; i < kFirTaps; ++i) low += fir_[i] * in_[c].at(start + i);
                    else
                        for (int i = 0; i < kFirTaps; ++i) {
                            const long long at = start + i;
                            if (at >= 0 && at < inCount_) low += fir_[i] * in_[c].at(at);
                        }
                    const double high = in_[c].at(filtered_) - low;
                    low_[c].at(filtered_) = low;
                    high_[c].at(filtered_) = high;
                    monoLow += low;
                    monoHigh += high;
                }
                lowMono_.at(filtered_) = monoLow / ch_;
                highMono_.at(filtered_) = monoHigh / ch_;
            }
            return;
        }
        const long long limit = inCount_ - kFirTaps;
        const double* h = fir_.data();
        for (; filtered_ <= limit; ++filtered_) {
            if (filtered_ + 3 <= limit) {
                const long long p = filtered_;
                for (int c = 0; c < ch_; ++c) {
                    const Ring<float>& r = in_[c];
                    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
                    for (int i = 0; i < kFirTaps; ++i) {
                        const double hi = h[i];
                        a0 += hi * r.at(p + i);
                        a1 += hi * r.at(p + 1 + i);
                        a2 += hi * r.at(p + 2 + i);
                        a3 += hi * r.at(p + 3 + i);
                    }
                    const double acc[4] = {a0, a1, a2, a3};
                    for (int k = 0; k < 4; ++k) {
                        low_[c].at(p + k) = acc[k];
                        high_[c].at(p + k) = r.at(p + k + kFirTaps / 2) - acc[k];
                    }
                }
                for (int k = 0; k < 4; ++k) {
                    double sl = 0.0, sh = 0.0;
                    for (int c = 0; c < ch_; ++c) {
                        sl += low_[c].at(p + k);
                        sh += high_[c].at(p + k);
                    }
                    lowMono_.at(p + k) = sl / ch_;
                    highMono_.at(p + k) = sh / ch_;
                }
                filtered_ += 3;
                continue;
            }
            for (int c = 0; c < ch_; ++c) {
                double acc = 0.0;
                for (int i = 0; i < kFirTaps; ++i) acc += fir_[i] * in_[c].at(filtered_ + i);
                low_[c].at(filtered_) = acc;
                // delayed original minus low band == high band, sample exact
                high_[c].at(filtered_) = in_[c].at(filtered_ + kFirTaps / 2) - acc;
            }
            // The alignment search compares a channel-averaged band against the
            // output, and it does so for hundreds of candidate positions per
            // grain. Averaging once per sample here instead of once per
            // candidate is the same number, and it is what brought the per-grain
            // burst inside the real-time budget: the old form did one divide per
            // channel per candidate, and divides are what the spike was made of.
            double sl = 0.0, sh = 0.0;
            for (int c = 0; c < ch_; ++c) {
                sl += low_[c].at(filtered_);
                sh += high_[c].at(filtered_);
            }
            lowMono_.at(filtered_) = sl / ch_;
            highMono_.at(filtered_) = sh / ch_;
        }
    }

    // ---- pitch tracking --------------------------------------------------
    // YIN: the cumulative mean normalised difference function, with parabolic
    // interpolation of the chosen lag. Public prior art, and cheap enough to
    // run per hop in a real-time path.
    // The difference function is the expensive part: one frame costs about
    // maxLag * frameLen multiply-adds, and doing a whole frame the instant its
    // audio arrives puts that entire cost in one block. Measured at a 128-sample
    // block that single burst was 108 % of the real-time budget while the
    // average was a tenth of it. The work is therefore SPREAD: each call does a
    // share of the lags proportional to the audio it was handed, so the cost per
    // block is flat. The arithmetic per lag is unchanged and the frame is
    // finished before any grain that needs it is placed, so the audio is
    // bit-identical to computing it all at once.
    void trackAvailable(int arrived) {
        long long budget = lagBudget(arrived) + (boundedScheduling_ ? lagRemainder_ : 0);
        lagRemainder_ = 0;
        for (;;) {
            if (!estimating_) {
                const long long start = static_cast<long long>(yinFrame_) * kYinHop;
                if (start + frameLen_ > inCount_) return;
                beginFrame(start);
            }
            const long long spent = stepFrame(budget);
            budget -= spent;
            if (estimating_) {
                if (boundedScheduling_) lagRemainder_ = budget;
                return;
            }
            finishFrame();
            ++yinFrame_;
            if (budget <= 0) return;
        }
    }

    // A frame's worth of lags per hop of audio, plus a little so the tracker
    // gains on the input rather than drifting behind it.
    long long lagBudget(int arrived) {
        if (drainTracker_) return 1LL << 40;  // finishing: cost no longer matters
        const long long perFrame = std::max<long long>(1, maxLag() - tMin_ + 1);
        // Proportional to the audio actually handed over, not rounded up to a
        // whole hop: rounding up let a 64-sample block pay for a whole frame,
        // which is precisely the burst this is meant to remove. A frame is due
        // every kYinHop samples, so 9/8 of that rate keeps the tracker gaining
        // on the input with margin to spare while staying smooth.
        if (boundedScheduling_) {
            lagFraction_ += perFrame * static_cast<long long>(arrived) * 9;
            const long long budget = lagFraction_ / (8 * kYinHop);
            lagFraction_ %= 8 * kYinHop;
            return budget;
        }
        return std::max<long long>(1, perFrame * static_cast<long long>(arrived) * 9 / (8 * kYinHop));
    }

    int maxLag() const { return std::min(tMax_, frameLen_ / 2); }

    void beginFrame(long long start) {
        const int n = frameLen_;
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int c = 0; c < ch_; ++c) s += in_[c].at(start + i);
            scratchX_[i] = s / ch_;
        }
        std::fill(scratchD_.begin(), scratchD_.end(), 0.0);
        curLag_ = tMin_;
        estimating_ = true;
    }

    // Returns how much of the budget it used; leaves estimating_ set if the
    // frame is not finished yet.
    long long stepFrame(long long budget) {
        const int n = frameLen_;
        const int hi = maxLag();
        const double* x = scratchX_.data();
        long long used = 0;
        while (curLag_ <= hi && used < budget) {
            const int l0 = curLag_;
            if (boundedScheduling_ && l0 + 3 <= hi && used + 4 > budget) break;
            if (l0 + 3 <= hi && used + 4 <= budget) {
                double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
                const int m = n - (l0 + 3);
                for (int i = 0; i < m; ++i) {
                    const double xi = x[i];
                    const double d0 = xi - x[i + l0];
                    const double d1 = xi - x[i + l0 + 1];
                    const double d2 = xi - x[i + l0 + 2];
                    const double d3 = xi - x[i + l0 + 3];
                    a0 += d0 * d0;
                    a1 += d1 * d1;
                    a2 += d2 * d2;
                    a3 += d3 * d3;
                }
                for (int i = m; i + l0 < n; ++i) {
                    const double d = x[i] - x[i + l0];
                    a0 += d * d;
                }
                for (int i = m; i + l0 + 1 < n; ++i) {
                    const double d = x[i] - x[i + l0 + 1];
                    a1 += d * d;
                }
                for (int i = m; i + l0 + 2 < n; ++i) {
                    const double d = x[i] - x[i + l0 + 2];
                    a2 += d * d;
                }
                scratchD_[l0] = a0;
                scratchD_[l0 + 1] = a1;
                scratchD_[l0 + 2] = a2;
                scratchD_[l0 + 3] = a3;
                curLag_ += 4;
                used += 4;
                continue;
            }
            double acc = 0.0;
            for (int i = 0; i + l0 < n; ++i) {
                const double diff = x[i] - x[i + l0];
                acc += diff * diff;
            }
            scratchD_[l0] = acc;
            ++curLag_;
            ++used;
        }
        if (curLag_ > hi) estimating_ = false;
        return used;
    }

    void finishFrame() {
        std::vector<double>& x = scratchX_;
        std::vector<double>& d = scratchD_;
        const int maxLag = this->maxLag();
        std::vector<double>& cmnd = scratchC_;
        std::fill(cmnd.begin(), cmnd.end(), 1.0);
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
        period_ = std::clamp(refined, static_cast<double>(tMin_), static_cast<double>(tMax_));
        record(true);

        // Excitation polarity: ONE sign for the whole stream, decided from the
        // very first analysis frame and never revised.
        //
        // It used to accumulate over the first eight voiced frames, and that was
        // a determinism hole: how many frames had been seen when the first mark
        // was refined depended on how the host chunked its feeds, so the sign
        // itself did -- and a flip lands every later mark half a period away.
        // Waiting for eight frames instead would have cost 43 ms of extra
        // start-up latency.
        //
        // Frame zero is enough because what matters is CONSISTENCY, not which
        // sign is picked: pitch marks only have to sit at the same phase of
        // every period. A "wrong" but constant choice places them on the other
        // side of the glottal pulse and splices just as cleanly.
        if (frames_ == 1) {
            double cube = 0.0;
            for (double v : x) cube += v * v * v;
            polarity_ = cube >= 0.0 ? 1.0 : -1.0;
        }
    }

    // The contour is stored per analysis frame and looked up BY POSITION. Using
    // "the newest estimate" instead makes a grain depend on how far tracking
    // happened to have run when the host called feed(), which is a second way
    // to lose block-size determinism.
    void record(bool voiced) {
        if (boundedScheduling_)
            sourcePhase_.at(frames_) = frames_ == 0 ? 0
                                                    : sourcePhase_.at(frames_ - 1) +
                                                          .5 * kYinHop * (1 / periods_.at(frames_ - 1) + 1 / period_);
        if (frames_ - metadataOldest_ >= metadataCapacity_) {
            metadataF0_.grow(metadataCapacity_ * 2, metadataOldest_, frames_);
            metadataCapacity_ *= 2;
        }
        metadataF0_.at(frames_) = voiced ? double(sr_) / period_ : 0;
        periods_.at(frames_) = period_;
        voicedFlags_.at(frames_) = voiced ? 1 : 0;
        ++frames_;
    }

    long long idealFrame(long long pos) const { return std::llround((pos - frameLen_ / 2.0) / kYinHop); }

    long long frameFor(long long pos) const {
        if (frames_ == 0) return -1;
        long long idx = idealFrame(pos);
        // The ring only holds the recent contour; a position older than that
        // cannot be rendered any more, so clamping to what is live is the same
        // answer a full history would give for any reachable position.
        const long long oldest = std::max<long long>(0, frames_ - ctrCap_);
        return std::clamp<long long>(idx, oldest, frames_ - 1);
    }

    double periodAt(long long pos) const {
        const long long i = frameFor(pos);
        return i < 0 ? period_ : periods_.at(i);
    }

    bool voicedAt(long long pos) const {
        const long long i = frameFor(pos);
        return i < 0 ? false : voicedFlags_.at(i) != 0;
    }

    // ---- synthesis -------------------------------------------------------
    // How much filtered input must exist before a grain centred at `centre`
    // can be placed, including its search range and its own half length.
    // Output position -> source position. Anchored, so a ratio change bends
    // the mapping from that point on instead of rewriting the past. With no
    // ratio change the anchors are zero and this is exactly tOut_ / alpha_.
    long long mapSource() const {
        return srcAnchor_ + static_cast<long long>(std::llround((tOut_ - outAnchor_) / alpha_));
    }

    long long needFor(long long centre) const {
        // Everything a grain step may read: its own half length, the alignment
        // search, and the window in which the next pitch mark is refined. The
        // amount is FIXED rather than "whatever has arrived", because a rule
        // that depends on arrival makes the output depend on the host's block
        // size -- which is exactly how determinism was lost the first time.
        return centre + static_cast<long long>((1.0 + kVoicedTol + 0.3) * tMax_) + 4;
    }

    void run(int arrived) {
        filterAvailable();
        trackAvailable(arrived);
        const long long haveFiltered = filtered_;  // low_/high_ length
        if (!started_) {
            // Wait for the tracker as well as for the audio: starting before
            // the first f0 estimate exists would anchor the schedule on a
            // default period, and whether that happens depends on the host's
            // block size.
            if (!finished_ && (haveFiltered < needFor(tMax_) || frames_ == 0)) return;
            // Anchor on the FIRST tracked frame, not on whatever the tracker
            // had reached by the time enough input arrived: the latter shifts
            // the whole grain schedule when the host uses a different block.
            const double first = frames_ == 0 ? period_ : periods_.at(0);
            markPos_ = static_cast<long long>(std::llround(first));
            tOut_ = static_cast<double>(markPos_);
            started_ = true;
        }

        // The grain loop is a small state machine so the alignment search can
        // be spread across calls the same way the tracker is. One grain's
        // search is a few hundred thousand multiply-adds and grains are ~11 ms
        // apart, so paying for a whole one inside a single 64-sample block was
        // a 14x spike over the average. Splitting it changes nothing about
        // which offset is chosen -- candidates are evaluated in the same order
        // and compared the same way -- only when the work happens.
        long long budget = searchBudget(arrived);
        for (;;) {
            if (stage_ == Stage::Pick) {
                long long srcCentre = mapSource();
                while (!ratioChanges_.empty() && srcCentre >= ratioChanges_.front().at) {
                    const auto change = ratioChanges_.front();
                    ratioChanges_.pop_front();
                    const double output = outAnchor_ + (change.at - srcAnchor_) * alpha_;
                    srcAnchor_ = change.at;
                    outAnchor_ = output;
                    alpha_ = change.ratio;
                    srcCentre = mapSource();
                }
                if (!finished_ && needFor(srcCentre) > haveFiltered) break;
                // The contour entry this grain belongs to must already exist.
                // Without this, a tracker that is spreading its work could
                // still be a frame behind, frameFor() would clamp to an older
                // entry, and the grain would be built from a different period
                // depending purely on timing.
                if (!finished_ && idealFrame(srcCentre) >= frames_) break;
                if (finished_ && target_ > 0 && static_cast<long long>(std::llround(tOut_)) >= target_) break;
                if (srcCentre >= haveFiltered) {
                    if (!finished_) break;
                    // Draining: keep laying grains from the last real audio so
                    // the requested length is reached without a truncated final
                    // period.
                    srcCentre = haveFiltered - 1;
                    if (srcCentre < 0) break;
                }

                // Freeze how much input this grain may look at. The search is
                // spread over several calls, so more audio arrives while it
                // runs; letting later candidates see further than earlier ones
                // makes the chosen offset depend on the host's block size.
                gFiltered_ = haveFiltered;
                gSrc_ = srcCentre;
                gT_ = periodAt(srcCentre);
                gL_ = std::max(2, static_cast<int>(std::lround(gT_)));
                gVo_ = voicedAt(srcCentre);
                gCentre_ = gVo_ ? nearestMark(srcCentre, gT_) : srcCentre;
                gHarmonicAverage_ = boundedScheduling_ && gVo_;
                gProjectionReady_ = false;
                gHarmonicFrequency_ = 1 / gT_;
                gHarmonicSlope_ = (gHarmonicFrequency_ - 1 / periodAt(std::max(0LL, gSrc_ - kYinHop))) / kYinHop;
                if (std::abs(gHarmonicSlope_) > .05 * gHarmonicFrequency_ * gHarmonicFrequency_)
                    gHarmonicAverage_ = false;
                if (gHarmonicAverage_) {
                    const long long last = frameFor(gSrc_);
                    gSourceFirstFrame_ = std::max(0LL, last - static_cast<long long>(gSourceFrequency_.size()) + 1);
                    gSourceModels_ = int(last - gSourceFirstFrame_ + 1);
                    for (int model = 0; model < gSourceModels_; ++model) {
                        const long long frame = gSourceFirstFrame_ + model;
                        gSourceFrequency_[model] = 1 / periods_.at(frame);
                        gSourcePhase_[model] = sourcePhase_.at(frame);
                        gSourceVoiced_[model] = voicedFlags_.at(frame) != 0;
                    }
                    gSourceFirstCenter_ = gSourceFirstFrame_ * kYinHop + frameLen_ / 2;
                    double currentEnergy = 0;
                    for (long long p = std::max(0LL, gCentre_ - gL_); p < std::min(gFiltered_, gCentre_ + gL_); ++p)
                        currentEnergy += lowMono_.at(p) * lowMono_.at(p);
                    for (int cycle = 0; cycle < 8; ++cycle) {
                        gCycleGain_[cycle] = 0;
                        gCycleShift_[cycle] = 0;
                        gCycleHighGain_[cycle] = cycle == 0 ? 1 : 0;
                        const double center = cycle == 0 ? double(gCentre_) : pastCyclePosition(gCentre_, cycle);
                        if (center < 0) continue;
                        const int frame = std::clamp(int(std::lround((center - gSourceFirstCenter_) / kYinHop)), 0,
                                                     gSourceModels_ - 1);
                        if (cycle > 0 && !gSourceVoiced_[frame]) continue;
                        const long long first = std::max(0LL, static_cast<long long>(std::lround(center)) - gL_);
                        const long long end = std::min(gFiltered_, static_cast<long long>(std::lround(center)) + gL_);
                        double energy = 0;
                        for (long long p = first; p < end; ++p) energy += lowMono_.at(p) * lowMono_.at(p);
                        if (cycle == 0)
                            gCycleGain_[cycle] = 1;
                        else if (energy > 1e-20 && energy >= .5 * currentEnergy && energy <= 2 * currentEnergy)
                            gCycleGain_[cycle] = std::sqrt(currentEnergy / energy);
                    }
                    for (int cycle = 1; cycle < 8; ++cycle) {
                        if (gCycleGain_[cycle] <= 0) continue;
                        for (int i = 0; i < 2 * gL_; i += 4)
                            gMatchSource_[i] = pastCyclePosition(double(gCentre_ - gL_ + i), cycle);
                        const int radius = std::max(2, int(.35 * gT_));
                        bool interior = gCentre_ - gL_ >= 0 && gCentre_ + gL_ <= gFiltered_;
                        double referenceEnergy = 0;
                        for (int i = 0; i < 2 * gL_; i += 4) {
                            const double position = gMatchSource_[i];
                            const long long index = static_cast<long long>(std::floor(position));
                            const double t = position - index, t2 = t * t, t3 = t2 * t;
                            gMatchIndex_[i] = index;
                            gMatchBasis_[i] = {-.5 * t + t2 - .5 * t3, 1 - 2.5 * t2 + 1.5 * t3,
                                               .5 * t + 2 * t2 - 1.5 * t3, -.5 * t2 + .5 * t3};
                            interior = interior && index - radius - 3 >= 0 && index + radius + 4 < gFiltered_;
                            const long long at = gCentre_ - gL_ + i;
                            gMatchReference_[i] = at >= 0 && at < gFiltered_ ? lowMono_.at(at) : 0;
                            referenceEnergy += gMatchReference_[i] * gMatchReference_[i];
                        }

                        auto sample = [&](double position) {
                            const long long at = static_cast<long long>(std::floor(position));
                            const double t = position - at;
                            auto get = [&](long long p) { return p < 0 || p >= gFiltered_ ? 0.0 : lowMono_.at(p); };
                            const double a = get(at - 1), b = get(at), c = get(at + 1), d = get(at + 2);
                            return b + .5 * t * (c - a + t * (2 * a - 5 * b + 4 * c - d + t * (3 * (b - c) + d - a)));
                        };
                        auto score = [&](int shift) {
                            double dot = 0, energy = 0;
                            if (interior) {
                                for (int i = 0; i < 2 * gL_; i += 4) {
                                    const long long at = gMatchIndex_[i] + shift;
                                    const auto& w = gMatchBasis_[i];
                                    const double b = w[0] * lowMono_.at(at - 1) + w[1] * lowMono_.at(at) +
                                                     w[2] * lowMono_.at(at + 1) + w[3] * lowMono_.at(at + 2);
                                    dot += gMatchReference_[i] * b;
                                    energy += b * b;
                                }
                                return dot / std::sqrt(energy * referenceEnergy + 1e-30);
                            }
                            double reference = 0;
                            for (int i = 0; i < 2 * gL_; i += 4) {
                                const long long at = gCentre_ - gL_ + i;
                                if (at < 0 || at >= gFiltered_) continue;
                                const double a = lowMono_.at(at), b = sample(gMatchSource_[i] + shift);
                                dot += a * b;
                                energy += b * b;
                                reference += a * a;
                            }
                            return dot / std::sqrt(energy * reference + 1e-30);
                        };
                        int best = 0;
                        double similarity = score(0);
                        for (int shift = -radius; shift <= radius; shift += 2) {
                            const double value = score(shift);
                            if (value > similarity) {
                                similarity = value;
                                best = shift;
                            }
                        }
                        if (similarity < .8) {
                            gCycleGain_[cycle] = 0;
                            continue;
                        }
                        const double left = score(best - 2), right = score(best + 2),
                                     den = left - 2 * similarity + right;
                        gCycleShift_[cycle] = best + (den < -1e-9 ? std::clamp((left - right) / den, -1.0, 1.0) : 0);
                    }
                    const int range = int(std::ceil((1 + kVoicedTol) * gT_)) + 4;
                    gHarmonicFirst_ = std::max(0LL, gCentre_ - range);
                    const long long end = std::min(gFiltered_, gCentre_ + range + 1);
                    gHarmonicCount_ = int(end - gHarmonicFirst_);
                    for (int i = 0; i < gHarmonicCount_; ++i) {
                        const long long position = gHarmonicFirst_ + i;
                        const double phase = sourcePhaseAt(double(position));
                        for (int cycle = 1; cycle < 8; ++cycle) {
                            const size_t slot = size_t(i) * 8 + cycle;
                            gProjectionIndex_[slot] = -1;
                            if (gCycleGain_[cycle] <= 0) continue;
                            const double source = positionAtSourcePhase(phase - cycle) + gCycleShift_[cycle];
                            if (source < 0) continue;
                            const long long index = static_cast<long long>(std::floor(source));
                            gProjectionIndex_[slot] = index;
                            gProjectionFraction_[slot] = source - index;
                        }
                    }
                    gProjectionReady_ = true;
                    for (int i = 0; i < gHarmonicCount_; ++i)
                        harmonicMono_[i] = harmonicSource(lowMono_, gHarmonicFirst_ + i);
                }
                beginSearch(gCentre_, gL_, static_cast<int>((gVo_ ? kVoicedTol : kUnvoicedTol) * gT_));
                stage_ = Stage::SearchLow;
            }

            if (stage_ == Stage::SearchLow) {
                budget -= stepSearch(lowMono_, budget);
                if (searching_) return;
                addGrain(outLow_, low_, gCentre_ + gBestD_, gL_, tOut_, false, true);
                if (boundedScheduling_ && gVo_ && highCoherent(gCentre_ + gBestD_)) {
                    gHighBlend_ = std::clamp((highPeriodicity(gCentre_ + gBestD_) - .8) / .2, 0.0, 1.0);
                    addGrain(outHigh_, high_, gCentre_ + gBestD_, gL_, tOut_, false, false, true);
                    tOut_ += gT_;
                    ++cycle_;
                    publish();
                    stage_ = Stage::Pick;
                    if (budget <= 0 && !finished_) break;
                    continue;
                }
                // The high band's template is taken AFTER the low grain lands,
                // exactly as when the two searches ran back to back.
                gReverse_ = reverseAt(gSrc_);
                beginSearch(gSrc_, gL_, static_cast<int>(kUnvoicedTol * gT_), true);
                stage_ = Stage::SearchHigh;
            }

            if (stage_ == Stage::SearchHigh) {
                budget -= stepSearch(highMono_, budget);
                if (searching_) return;
                addGrain(outHigh_, high_, gSrc_ + gBestD_, gL_, tOut_,
                         boundedScheduling_ ? gReverse_ : reverseThisGrain(), false);
                tOut_ += gT_;
                ++cycle_;
                publish();
                stage_ = Stage::Pick;
            }
            if (budget <= 0 && !finished_) break;
        }
        publish();
    }

    // One mark per period, refined to the local extremum of the chosen
    // polarity; marks advance strictly forward so the schedule is causal.
    long long nearestMark(long long target, double T) {
        const int w = std::max(2, static_cast<int>(0.3 * T));
        // A long unvoiced stretch leaves markPos_ far behind: the walk below
        // steps one period at a time, so after a pause it would grind through
        // thousands of iterations reading band positions that the ring has long
        // since recycled -- i.e. from unrelated audio. Marks only matter
        // relative to the grain being placed, so skip the dead ground.
        const long long guard = 4LL * tMax_;
        if (target - markPos_ > guard) markPos_ = target - guard;
        while (markPos_ + static_cast<long long>(std::lround(T)) <= target) {
            long long guess = markPos_ + static_cast<long long>(std::lround(T));
            long long a = std::max<long long>(0, guess - w);
            long long b = std::min<long long>(gFiltered_, guess + w);
            if (b <= a) break;
            long long bestI = guess;
            double bestV = -1e30;
            for (long long i = a; i < b; ++i) {
                double s;
                s = lowMono_.at(i) * polarity_;
                if (s > bestV) {
                    bestV = s;
                    bestI = i;
                }
            }
            markPos_ = bestI;
        }
        return markPos_;
    }

    // WSOLA alignment, split into a setup and a resumable step. The template
    // is what has already been written around the placement point; candidates
    // shift the source grain, and the one whose leading half correlates best
    // wins, so the grain adds constructively instead of combing.
    void beginSearch(long long centre, int L, int tol, bool highBand = false) {
        gSearchHigh_ = highBand;
        gTol_ = tol;
        gSearchCentre_ = centre;
        gBestD_ = 0;
        gBestC_ = -1e30;
        gGrainL_ = L;
        searching_ = false;
        if (tol < 2) return;
        const long long tb = static_cast<long long>(std::llround(tOut_));
        const long long ta = std::max<long long>(0, tb - L);
        gM_ = static_cast<int>(tb - ta);
        if (gM_ < 8) return;
        double energy = 0.0;
        for (int i = 0; i < gM_; ++i) {
            const long long idx = ta + i;
            double v = 0;
            if (boundedScheduling_) {
                const double weight = idx < written_ ? (highBand ? highNorm_.at(idx) : norm_.at(idx)) : 0;
                if (weight > 1e-6) {
                    for (int c = 0; c < ch_; ++c) v += highBand ? outHigh_[c].at(idx) : outLow_[c].at(idx);
                    v /= (ch_ * weight);
                }
            } else {
                const double w = idx < written_ ? norm_.at(idx) : 0.0;

                const double hw = idx < written_ ? highNorm_.at(idx) : 0;
                if (!boundedScheduling_ || hw == w) {
                    if (w > 1e-6) {
                        for (int c = 0; c < ch_; ++c) v += outLow_[c].at(idx) + outHigh_[c].at(idx);
                        v /= (ch_ * w);
                    }
                } else {
                    for (int c = 0; c < ch_; ++c)
                        v += outLow_[c].at(idx) / (w > 1e-6 ? w : 1.0) + outHigh_[c].at(idx) / (hw > 1e-6 ? hw : 1.0);
                    v /= ch_;
                }
            }
            scratchT_[i] = v;
            energy += v * v;
        }
        if (energy < 1e-18) return;
        gCand_ = -tol;
        searching_ = true;
    }

    long long stepSearch(Ring<double>& bandMono, long long budget) {
        if (!searching_) return 0;
        long long used = 0;
        while (gCand_ <= gTol_ && used < budget) {
            const long long cs = gSearchCentre_ + gCand_;
            const long long a = std::max<long long>(0, cs - gGrainL_);
            const int len = static_cast<int>(std::min<long long>(cs - a, gM_));
            if (len >= 8) {
                double dot = 0.0, nrm = 0.0;
                for (int i = 0; i < len; ++i) {
                    const long long end = std::min(gFiltered_, cs + gGrainL_);
                    const long long si =
                        boundedScheduling_ && gSearchHigh_ && gReverse_ ? a + end - 1 - cs + len - i : cs - len + i;
                    if (si < 0 || si >= gFiltered_) continue;
                    double sv = bandMono.at(si);
                    if (gHarmonicAverage_ && !gSearchHigh_) {
                        const long long offset = si - gHarmonicFirst_;
                        sv = offset >= 0 && offset < gHarmonicCount_ ? harmonicMono_[size_t(offset)]
                                                                     : harmonicSource(bandMono, si);
                    }
                    dot += sv * scratchT_[gM_ - len + i];
                    nrm += sv * sv;
                }
                const double c = dot / (std::sqrt(nrm) + 1e-9);
                if (c > gBestC_) {
                    gBestC_ = c;
                    gBestD_ = static_cast<int>(gCand_);
                }
            }
            gCand_ += 2;
            ++used;
        }
        if (gCand_ > gTol_) searching_ = false;
        return used;
    }

    // Candidates one call may evaluate, proportional to the audio it was given.
    long long searchBudget(int arrived) const {
        if (finished_) return 1LL << 40;
        // A grain costs (0.9 + 0.4) periods of candidates at a step of two and
        // buys one period of output: about 1.3 candidates per output sample.
        // The budget MUST exceed that. Setting it to 1.0 looked smoother and
        // was a throughput deficit -- synthesis fell a little further behind
        // every block, until the source position it still needed had been
        // recycled out of the band ring and it started correlating against
        // unrelated audio. Two per output sample leaves 50 % of headroom.
        const double outPerIn = std::max(alpha_, 0.05);
        return std::max<long long>(8, static_cast<long long>(arrived * outPerIn * 2.0));
    }

    bool highCoherent(long long end) const { return highPeriodicity(end) >= .8; }
    double highPeriodicity(long long end) const {
        const int length = std::max(8, int(std::lround(gT_)));
        double cross = 0, a2 = 0, b2 = 0;
        for (int i = 0; i < length; ++i) {
            const long long at = end - length + i;
            const double lagged = at - gT_;
            const long long low = static_cast<long long>(std::floor(lagged));
            if (low < 0 || at < 0 || at >= gFiltered_) continue;
            const double fraction = lagged - low;
            const double a = highMono_.at(at);
            const double b = highMono_.at(low) * (1 - fraction) + highMono_.at(low + 1) * fraction;
            cross += a * b;
            a2 += a * a;
            b2 += b * b;
        }
        if (a2 > 1e-18 && b2 > 1e-18) return std::clamp(cross / std::sqrt(a2 * b2), -1.0, 1.0);
        return false;
    }

    bool reverseThisGrain() const { return reverseAt(gSrc_ + gBestD_); }
    bool reverseAt(long long center) const {
        if (boundedScheduling_ && gVo_ && highCoherent(center)) return false;
        // Reverse one grain in every `p`, where p tracks the reuse count, i.e.
        // the stretch ratio. Measured across 1.25x-4x the best setting is
        // always p = round(alpha): at 2x plain alternation, at 3x every third,
        // at 4x every fourth. Getting this wrong is not a small loss -- a
        // mismatched period (every third grain at 2x) fills the spectral
        // valleys by 3.5 dB and is rejected outright.
        const int p = revPeriod_ > 0 ? revPeriod_ : std::clamp(static_cast<int>(std::lround(alpha_)), 2, 8);
        return (cycle_ % p) == 1;
    }

    double sourcePhaseAt(double position) const {
        const double relative = (position - gSourceFirstCenter_) / kYinHop;
        if (relative < 0) return gSourcePhase_[0] + gSourceFrequency_[0] * (position - gSourceFirstCenter_);
        const int model = std::clamp(int(std::floor(relative)), 0, gSourceModels_ - 1);
        const double delta = position - (gSourceFirstCenter_ + model * kYinHop);
        const double slope = model + 1 < gSourceModels_
                                 ? (gSourceFrequency_[model + 1] - gSourceFrequency_[model]) / kYinHop
                                 : gHarmonicSlope_;
        return gSourcePhase_[model] + gSourceFrequency_[model] * delta + .5 * slope * delta * delta;
    }
    double pastCyclePosition(double position, int cycle) const {
        return positionAtSourcePhase(sourcePhaseAt(position) - cycle);
    }
    double positionAtSourcePhase(double phase) const {
        if (phase < gSourcePhase_[0]) return gSourceFirstCenter_ + (phase - gSourcePhase_[0]) / gSourceFrequency_[0];
        const auto upper = std::upper_bound(gSourcePhase_.begin(), gSourcePhase_.begin() + gSourceModels_, phase);
        const int model = std::clamp(int(upper - gSourcePhase_.begin()) - 1, 0, gSourceModels_ - 1);
        const double delta = phase - gSourcePhase_[model], frequency = gSourceFrequency_[model];
        const double slope =
            model + 1 < gSourceModels_ ? (gSourceFrequency_[model + 1] - frequency) / kYinHop : gHarmonicSlope_;
        const double discriminant = frequency * frequency + 2 * slope * delta;
        if (discriminant <= 0) return -1;
        return gSourceFirstCenter_ + model * kYinHop + 2 * delta / (frequency + std::sqrt(discriminant));
    }
    double harmonicSource(const Ring<double>& band, long long position, bool /*highBand*/ = false) const {
        double sum = band.at(position), sx = 0, sxx = 0, sxy = 0;
        int count = 1;
        for (int cycle = 1; cycle < 8; ++cycle) {
            const double gain = gCycleGain_[cycle];
            if (gain <= 0) continue;
            long long index;
            double t;
            const long long offset = position - gHarmonicFirst_;
            if (gProjectionReady_ && offset >= 0 && offset < gHarmonicCount_) {
                const size_t slot = size_t(offset) * 8 + cycle;
                index = gProjectionIndex_[slot];
                if (index < 0) continue;
                t = gProjectionFraction_[slot];
            } else {
                const double source = pastCyclePosition(double(position), cycle) + gCycleShift_[cycle];
                if (source < 0) continue;
                index = static_cast<long long>(std::floor(source));
                t = source - index;
            }
            auto sample = [&](long long p) { return p < 0 ? 0.0 : band.at(p); };
            const double a = sample(index - 1), b = sample(index), c = sample(index + 1), d = sample(index + 2);
            const double value =
                gain * (b + .5 * t * (c - a + t * (2 * a - 5 * b + 4 * c - d + t * (3 * (b - c) + d - a))));
            sum += value;
            sx += cycle;
            sxx += cycle * cycle;
            sxy += cycle * value;
            ++count;
        }
        const double determinant = count * sxx - sx * sx;
        return determinant > 0 ? (sxx * sum - sx * sxy) / determinant : sum / count;
    }
    void addGrain(std::vector<Ring<double>>& dst, std::vector<Ring<double>>& band, long long centre, int L, double tOut,
                  bool reverse, bool accumulateNorm, bool averageHigh = false) {
        const long long a = centre - L, b = centre + L;
        const long long ga = std::max<long long>(0, a);
        const long long gb = std::min<long long>(gFiltered_, b);
        const int len = static_cast<int>(gb - ga);
        if (len < 4) return;
        const long long oa = static_cast<long long>(std::llround(tOut)) - (centre - ga);
        growOut(oa + len);
        for (int i = 0; i < len; ++i) {
            const long long oi = oa + i;
            if (oi < 0) continue;
            const double w = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / std::max(len - 1, 1));
            const int src = reverse ? (len - 1 - i) : i;
            for (int c = 0; c < ch_; ++c) {
                const double raw = band[c].at(ga + src);
                double value = raw;
                if ((accumulateNorm || averageHigh) && gHarmonicAverage_) {
                    const double harmonic = harmonicSource(band[c], ga + src);
                    value = averageHigh ? raw + gHighBlend_ * (harmonic - raw) : harmonic;
                }
                dst[c].at(oi) += value * w;
            }
            if (accumulateNorm)
                norm_.at(oi) += w;
            else
                highNorm_.at(oi) += w;
        }
    }

    // Zero every output slot up to `upto` the first time it is used, which is
    // what resizing a vector did implicitly, and remember how far that is.
    void growOut(long long upto) {
        if (upto <= written_) return;
        // Output the host has not read yet still has to be there when it asks.
        if (upto - outRead_ > outCap_) {
            const long long next = pow2(2 * (upto - outRead_));
            for (int c = 0; c < ch_; ++c) {
                outLow_[c].grow(next, outRead_, written_);
                outHigh_[c].grow(next, outRead_, written_);
            }
            norm_.grow(next, outRead_, written_);
            highNorm_.grow(next, outRead_, written_);
            outCap_ = next;
        }
        for (int c = 0; c < ch_; ++c) {
            outLow_[c].clearThrough(upto);
            outHigh_[c].clearThrough(upto);
        }
        norm_.clearThrough(upto);
        highNorm_.clearThrough(upto);
        written_ = upto;
    }

    // A sample is final once no later grain can reach it: grains are placed at
    // tOut_ and reach half a grain back, so everything before tOut_ - tMax_ is
    // settled.
    void publish() {
        long long safe = finished_ ? target_ : static_cast<long long>(std::llround(tOut_)) - tMax_;
        if (finished_)
            growOut(target_);
        else
            safe = std::min<long long>(safe, written_);
        if (safe > outWritten_) outWritten_ = safe;
    }

    int sr_ = 48000, ch_ = 1;
    int tMin_ = 96, tMax_ = 600, frameLen_ = 2048;
    double alpha_ = 1.0;
    std::vector<double> fir_;
    std::vector<Ring<float>> in_;
    std::vector<Ring<double>> low_, high_, outLow_, outHigh_;
    Ring<double> lowMono_, highMono_, norm_, highNorm_;
    Ring<double> periods_;
    Ring<int> voicedFlags_;
    std::vector<double> scratchX_, scratchD_, scratchC_, scratchT_;
    Ring<double> metadataF0_;
    long long metadataCapacity_ = 0, metadataOldest_ = 0;
    long long inCap_ = 0, bandCap_ = 0, outCap_ = 0, ctrCap_ = 0;
    long long written_ = 0, frames_ = 0;
    long long curLag_ = 0;
    bool estimating_ = false, drainTracker_ = false;
    struct RatioChange {
        long long at;
        double ratio;
    };
    std::deque<RatioChange> ratioChanges_;
    Ring<double> sourcePhase_;
    long long gSourceFirstFrame_ = 0, gSourceFirstCenter_ = 0;
    int gSourceModels_ = 0;
    std::vector<double> gSourceFrequency_, gSourcePhase_;
    std::vector<unsigned char> gSourceVoiced_;
    std::array<double, 8> gCycleGain_{}, gCycleShift_{}, gCycleHighGain_{};
    std::vector<double> gMatchSource_, gMatchReference_;
    std::vector<long long> gMatchIndex_;
    std::vector<std::array<double, 4>> gMatchBasis_;
    bool gHarmonicAverage_ = false;
    double gHighBlend_ = 0;
    double gHarmonicFrequency_ = 0, gHarmonicSlope_ = 0;
    long long gHarmonicFirst_ = 0;
    int gHarmonicCount_ = 0;
    std::vector<double> harmonicMono_, gProjectionFraction_;
    std::vector<long long> gProjectionIndex_;
    bool gProjectionReady_ = false;
    bool boundedScheduling_ = false, gSearchHigh_ = false, gReverse_ = false;
    long long lagFraction_ = 0, lagRemainder_ = 0;
    enum class Stage { Pick, SearchLow, SearchHigh };
    Stage stage_ = Stage::Pick;
    bool searching_ = false;
    long long gSrc_ = 0, gCentre_ = 0, gSearchCentre_ = 0, gCand_ = 0;
    long long gFiltered_ = 0;
    int gTol_ = 0, gM_ = 0, gL_ = 0, gGrainL_ = 0, gBestD_ = 0;
    double gT_ = 0.0, gBestC_ = 0.0;
    bool gVo_ = false;
    long long inCount_ = 0, filtered_ = 0, markPos_ = -1;
    long long outWritten_ = 0, outRead_ = 0, target_ = 0;
    int yinFrame_ = 0, cycle_ = 0;
    double tOut_ = 0.0, period_ = 320.0, outAnchor_ = 0.0;
    long long srcAnchor_ = 0;
    bool voiced_ = false, finished_ = false, started_ = false;
    double polarity_ = 1.0;
    int revPeriod_ = 0;
};

}  // namespace pbshift

#endif  // PBSHIFT_SPEECH_TIME_H
