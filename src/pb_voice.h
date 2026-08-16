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
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <vector>

namespace pbshift {

// Fixed-capacity ring addressed by ABSOLUTE sample position. Storage only --
// every read and write keeps the same value and the same order as a plain
// growing vector would, so swapping one in cannot change a single output
// sample. What it does change is that a stream can run for hours: a growing
// vector held ~170 MB after 30 seconds and reallocated as it grew, and that
// reallocation, not the arithmetic, is what blew the real-time budget (a
// 15.3 ms worst-case block against a 10.7 ms budget, on a 1.5 ms average).
template <typename T>
class Ring {
public:
    void init(long long capacityPow2) {
        cap_ = capacityPow2;
        mask_ = cap_ - 1;
        buf_.assign(static_cast<size_t>(cap_), T{});
        cleared_ = 0;
    }
    void clear() {
        std::fill(buf_.begin(), buf_.end(), T{});
        cleared_ = 0;
    }
    T& at(long long i) { return buf_[static_cast<size_t>(i & mask_)]; }
    const T& at(long long i) const {
        return buf_[static_cast<size_t>(i & mask_)];
    }
    // Accumulation buffers must start from zero at a position the first time it
    // is written, exactly as a freshly resized vector would.
    void clearThrough(long long upto) {
        while (cleared_ < upto) {
            buf_[static_cast<size_t>(cleared_ & mask_)] = T{};
            ++cleared_;
        }
    }
    long long capacity() const { return cap_; }

    // Grow while keeping the live window [from, to) at the same absolute
    // positions. Only reached when a host buffers far more output than it
    // reads; a host that reads each block never triggers it, so the real-time
    // path stays allocation-free.
    void grow(long long newCap, long long from, long long to) {
        std::vector<T> next(static_cast<size_t>(newCap), T{});
        const long long nmask = newCap - 1;
        for (long long i = from; i < to; ++i)
            next[static_cast<size_t>(i & nmask)] = buf_[static_cast<size_t>(i & mask_)];
        buf_.swap(next);
        cap_ = newCap;
        mask_ = nmask;
        if (cleared_ < to) cleared_ = to;
    }

private:
    std::vector<T> buf_;
    long long cap_ = 0, mask_ = 0, cleared_ = 0;
};

class VoiceStretch {
public:
    // Pitch search range. 80 Hz rather than 70 keeps the longest period (and
    // therefore the grain look-ahead) inside the low-latency budget; below this
    // the tracker falls back to holding the last period it trusted.
    static constexpr double kFmin = 80.0;
    static constexpr double kFmax = 500.0;
    static constexpr double kCrossoverHz = 3000.0;
    static constexpr int kYinFrameMin = 2048;
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
        // The frame has to hold at least two of the longest period the tracker
        // is allowed to find, or the search silently stops short of kFmin and
        // the pitch floor rises with the sample rate: a fixed 2048 caps the
        // search at 1024 lags, which is only 94 Hz at 96 kHz. Rounded up to a
        // power of two, this is exactly 2048 at 44.1 and 48 kHz, so those rates
        // render bit-identically to before.
        frameLen_ = static_cast<int>(
            pow2(std::max<long long>(kYinFrameMin, 3LL * tMax_)));
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
        periods_.init(ctrCap_);
        voicedFlags_.init(ctrCap_);

        // Every working buffer is sized once here. Nothing in feed() or read()
        // allocates, so a real-time thread never waits on the allocator.
        scratchX_.assign(frameLen_, 0.0);
        scratchD_.assign(tMax_ + 2, 0.0);
        scratchC_.assign(tMax_ + 2, 0.0);
        scratchT_.assign(2 * tMax_ + 8, 0.0);
        reset();
    }

    static long long pow2(long long n) {
        long long p = 1;
        while (p < n) p <<= 1;
        return p;
    }

    void reset() {
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
    int inputLatency() const {
        return frameLen_ / 2 + static_cast<int>((1.0 + kVoicedTol) * tMax_) +
               kFirTaps / 2;
    }
    // Output cannot be handed over until no later grain can still touch it: a
    // grain reaches half its length past its placement.
    int outputLatency() const { return inputLatency() + tMax_; }

    // The whole range the engine can hand over, not the range this engine was
    // tuned on. A narrower clamp here is silent and breaks the exact-length
    // guarantee: Voice + 4x + a pitch shift drove alpha to 8, the clamp made it
    // 4, and the post-resampler then halved that -- the host asked for 4x and
    // got 2x, with the whole stream off its timeline rather than just the tail.
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
    void setReversePeriod(int p) { revPeriod_ = p; }   // 0 = automatic

    void feed(const float* const* in, int frames) {
        // Consume in chunks no larger than half the input ring. A host is
        // allowed to hand over an entire file in one call, and writing that
        // straight in would overwrite samples the crossover has not read yet.
        const int chunk = static_cast<int>(std::max<long long>(1, inCap_ / 2));
        int done = 0;
        while (done < frames) {
            const int k = std::min(chunk, frames - done);
            for (int c = 0; c < ch_; ++c)
                for (int i = 0; i < k; ++i)
                    in_[c].at(inCount_ + i) = in[c][done + i];
            inCount_ += k;
            done += k;
            run(k);
        }
    }

    void finish() {
        finished_ = true;
        // Draining: no budget applies any more, so let the tracker complete.
        drainTracker_ = true;
        // The requested length is exact: a host asking for 2x expects twice the
        // samples it fed, not twice minus whatever the last grain happened to
        // reach.
        target_ = static_cast<long long>(std::llround(inCount_ * alpha_));
        run(frameLen_);
    }

    int available() const {
        return static_cast<int>(std::max<long long>(0, outWritten_ - outRead_));
    }

    int read(float* const* out, int frames) {
        const int n = std::min(frames, available());
        for (int c = 0; c < ch_; ++c) {
            for (int i = 0; i < n; ++i) {
                const long long idx = outRead_ + i;
                const double w = norm_.at(idx);
                out[c][i] = static_cast<float>(
                    (outLow_[c].at(idx) + outHigh_[c].at(idx)) /
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
                for (int i = 0; i < kFirTaps; ++i)
                    acc += fir_[i] * in_[c].at(filtered_ + i);
                low_[c].at(filtered_) = acc;
                // delayed original minus low band == high band, sample exact
                high_[c].at(filtered_) =
                    in_[c].at(filtered_ + kFirTaps / 2) - acc;
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
        long long budget = lagBudget(arrived);
        for (;;) {
            if (!estimating_) {
                const long long start =
                    static_cast<long long>(yinFrame_) * kYinHop;
                if (start + frameLen_ > inCount_) return;
                beginFrame(start);
            }
            const long long spent = stepFrame(budget);
            budget -= spent;
            if (estimating_) return;          // ran out of budget mid-frame
            finishFrame();
            ++yinFrame_;
            if (budget <= 0) return;
        }
    }

    // A frame's worth of lags per hop of audio, plus a little so the tracker
    // gains on the input rather than drifting behind it.
    long long lagBudget(int arrived) const {
        if (drainTracker_) return 1LL << 40;   // finishing: cost no longer matters
        const long long perFrame = std::max<long long>(1, maxLag() - tMin_ + 1);
        // Proportional to the audio actually handed over, not rounded up to a
        // whole hop: rounding up let a 64-sample block pay for a whole frame,
        // which is precisely the burst this is meant to remove. A frame is due
        // every kYinHop samples, so 9/8 of that rate keeps the tracker gaining
        // on the input with margin to spare while staying smooth.
        return std::max<long long>(
            1, perFrame * static_cast<long long>(arrived) * 9 / (8 * kYinHop));
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
        long long used = 0;
        while (curLag_ <= hi && used < budget) {
            double acc = 0.0;
            for (int i = 0; i + curLag_ < n; ++i) {
                const double diff = scratchX_[i] - scratchX_[i + curLag_];
                acc += diff * diff;
            }
            scratchD_[curLag_] = acc;
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
        period_ = std::clamp(refined, static_cast<double>(tMin_),
                             static_cast<double>(tMax_));
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
        periods_.at(frames_) = period_;
        voicedFlags_.at(frames_) = voiced ? 1 : 0;
        ++frames_;
    }

    long long idealFrame(long long pos) const {
        return std::llround((pos - frameLen_ / 2.0) / kYinHop);
    }

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
        return srcAnchor_ +
               static_cast<long long>(std::llround((tOut_ - outAnchor_) / alpha_));
    }

    long long needFor(long long centre) const {
        // Everything a grain step may read: its own half length, the alignment
        // search, and the window in which the next pitch mark is refined. The
        // amount is FIXED rather than "whatever has arrived", because a rule
        // that depends on arrival makes the output depend on the host's block
        // size -- which is exactly how determinism was lost the first time.
        return centre +
               static_cast<long long>((1.0 + kVoicedTol + 0.3) * tMax_) + 4;
    }

    void run(int arrived) {
        filterAvailable();
        trackAvailable(arrived);
        const long long haveFiltered = filtered_;   // low_/high_ length
        if (!started_) {
            // Wait for the tracker as well as for the audio: starting before
            // the first f0 estimate exists would anchor the schedule on a
            // default period, and whether that happens depends on the host's
            // block size.
            if (!finished_ && (haveFiltered < needFor(tMax_) || frames_ == 0))
                return;
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
                if (!finished_ && needFor(srcCentre) > haveFiltered) break;
                // The contour entry this grain belongs to must already exist.
                // Without this, a tracker that is spreading its work could
                // still be a frame behind, frameFor() would clamp to an older
                // entry, and the grain would be built from a different period
                // depending purely on timing.
                if (!finished_ && idealFrame(srcCentre) >= frames_) break;
                if (finished_ && target_ > 0 &&
                    static_cast<long long>(std::llround(tOut_)) >= target_)
                    break;
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
                beginSearch(gCentre_, gL_,
                            static_cast<int>((gVo_ ? kVoicedTol : kUnvoicedTol) *
                                             gT_));
                stage_ = Stage::SearchLow;
            }

            if (stage_ == Stage::SearchLow) {
                budget -= stepSearch(lowMono_, budget);
                if (searching_) return;
                addGrain(outLow_, low_, gCentre_ + gBestD_, gL_, tOut_, false,
                         true);
                // The high band's template is taken AFTER the low grain lands,
                // exactly as when the two searches ran back to back.
                beginSearch(gSrc_, gL_, static_cast<int>(kUnvoicedTol * gT_));
                stage_ = Stage::SearchHigh;
            }

            if (stage_ == Stage::SearchHigh) {
                budget -= stepSearch(highMono_, budget);
                if (searching_) return;
                addGrain(outHigh_, high_, gSrc_ + gBestD_, gL_, tOut_,
                         reverseThisGrain(), false);
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
    void beginSearch(long long centre, int L, int tol) {
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
            const double w = idx < written_ ? norm_.at(idx) : 0.0;
            double v = 0.0;
            if (w > 1e-6) {
                for (int c = 0; c < ch_; ++c)
                    v += outLow_[c].at(idx) + outHigh_[c].at(idx);
                v /= (ch_ * w);
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
                    const long long si = cs - len + i;
                    if (si < 0 || si >= gFiltered_) continue;
                    const double sv = bandMono.at(si);
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
        return std::max<long long>(
            8, static_cast<long long>(arrived * outPerIn * 2.0));
    }

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

    void addGrain(std::vector<Ring<double>>& dst, std::vector<Ring<double>>& band,
                  long long centre, int L, double tOut, bool reverse,
                  bool accumulateNorm) {
        const long long a = centre - L, b = centre + L;
        const long long ga = std::max<long long>(0, a);
        const long long gb = std::min<long long>(gFiltered_, b);
        const int len = static_cast<int>(gb - ga);
        if (len < 4) return;
        const long long oa =
            static_cast<long long>(std::llround(tOut)) - (centre - ga);
        growOut(oa + len);
        for (int i = 0; i < len; ++i) {
            const long long oi = oa + i;
            if (oi < 0) continue;
            const double w =
                0.5 - 0.5 * std::cos(2.0 * M_PI * i / std::max(len - 1, 1));
            const int src = reverse ? (len - 1 - i) : i;
            for (int c = 0; c < ch_; ++c)
                dst[c].at(oi) += band[c].at(ga + src) * w;
            if (accumulateNorm) norm_.at(oi) += w;
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
            outCap_ = next;
        }
        for (int c = 0; c < ch_; ++c) {
            outLow_[c].clearThrough(upto);
            outHigh_[c].clearThrough(upto);
        }
        norm_.clearThrough(upto);
        written_ = upto;
    }

    // A sample is final once no later grain can reach it: grains are placed at
    // tOut_ and reach half a grain back, so everything before tOut_ - tMax_ is
    // settled.
    void publish() {
        long long safe = finished_
                             ? target_
                             : static_cast<long long>(std::llround(tOut_)) -
                                   tMax_;
        if (finished_) growOut(target_);
        else safe = std::min<long long>(safe, written_);
        if (safe > outWritten_) outWritten_ = safe;
    }

    int sr_ = 48000, ch_ = 1;
    int tMin_ = 96, tMax_ = 600, frameLen_ = 2048;
    double alpha_ = 1.0;
    std::vector<double> fir_;
    std::vector<Ring<float>> in_;
    std::vector<Ring<double>> low_, high_, outLow_, outHigh_;
    Ring<double> lowMono_, highMono_, norm_;
    Ring<double> periods_;
    Ring<int> voicedFlags_;
    std::vector<double> scratchX_, scratchD_, scratchC_, scratchT_;
    long long inCap_ = 0, bandCap_ = 0, outCap_ = 0, ctrCap_ = 0;
    long long written_ = 0, frames_ = 0;
    long long curLag_ = 0;
    bool estimating_ = false, drainTracker_ = false;
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

#endif  // PBSHIFT_PB_VOICE_H
