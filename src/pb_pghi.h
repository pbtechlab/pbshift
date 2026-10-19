// Real-Time Phase Gradient Heap Integration (RTPGHI).
// Prusa & Sondergaard 2016/2017, implemented cleanroom from the papers.
//
// Given the previous synthesis frame's phases and both frames' measured
// phase gradients, integrates phases across the current frame: significant
// bins get heap-ordered trapezoidal integration (time direction from the
// previous frame, frequency direction from already-solved neighbours);
// bins below tolerance get deterministic position-seeded random phase.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>

#include "pb_stft.h"

namespace pbshift {

class Rtpghi {
public:
    explicit Rtpghi(int bins) : bins_(bins) {
        phase_.assign(bins, 0.0f);
        prevOmega_.assign(bins, 0.0f);
        prevMag_.assign(bins, 0.0f);
        xCand_.assign(bins, 0);
        xCandPrev_.assign(bins, 0);
        lazyY_.assign(bins, {0.0f, 0.0f});
        lazyFallback_.assign(bins, 0.0f);
        solved_.resize(bins);
        // at most one peak per two bins: sized once, so no step allocates
        peaks_.reserve(static_cast<size_t>(bins / 2 + 1));
        supPeaks_.reserve(static_cast<size_t>(bins / 2 + 1));
        havePrev_ = false;
    }

    void reset() {
        havePrev_ = false;
        lazy_ = false;
        std::fill(phase_.begin(), phase_.end(), 0.0f);
        std::fill(xCand_.begin(), xCand_.end(), 0);
        std::fill(xCandPrev_.begin(), xCandPrev_.end(), 0);
        xCandFrame_ = 0;
        // Per-stream: the tail treatment reads this at finish(); a value left
        // over from the previous stream made a reset() instance render a
        // short gliding stream differently from a fresh one.
        chirpFound_ = false;
    }

    // cur:    current analysis frame (mag/omega/tau filled)
    // hs:     synthesis hop in output samples
    // alpha:  local time-stretch ratio (scales group delay to output time)
    // frameIndex: for the deterministic RNG seed
    // resetMask: optional per-bin transient flags — those bins take the
    //            analysis phase verbatim (synchronized attack reset) and
    //            act as anchors for frequency-direction propagation
    // outPhase: resulting synthesis phases for this frame
    void step(const AnalysisFrame& cur, int hs, float alpha,
              uint64_t frameIndex, const uint8_t* resetMask,
              std::vector<float>& outPhase) {
        materialize();
        outPhase.resize(bins_);
        if (!havePrev_) {
            // first frame: copy analysis phases
            for (int m = 0; m < bins_; ++m)
                outPhase[m] = std::arg(cur.X[m]);
            finish(cur, outPhase);
            return;
        }

        float maxMag = 0.0f;
        for (int m = 0; m < bins_; ++m)
            if (cur.mag[m] > maxMag) maxMag = cur.mag[m];
        const float tol = tolRel_ * maxMag;

        std::fill(solved_.begin(), solved_.end(), 0u);
        int unsolved = 0;
        for (int m = 0; m < bins_; ++m) {
            if (resetMask && resetMask[m] && cur.mag[m] > tol) {
                outPhase[m] = std::arg(cur.X[m]);
                solved_[m] = 1;  // solved anchor
            } else if (cur.mag[m] > tol) {
                ++unsolved;
            } else {
                solved_[m] = 2;  // below tolerance -> random phase later
            }
        }

        // heap over (magnitude, bin, frame) — frame 0 = previous, 1 = current
        struct Entry {
            float mag;
            int bin;
            uint8_t frame;
            bool operator<(const Entry& o) const { return mag < o.mag; }
        };
        std::priority_queue<Entry> heap;
        for (int m = 0; m < bins_; ++m) {
            if (solved_[m] == 1)  // reset anchors propagate vertically
                heap.push({cur.mag[m], m, 1});
            else if (solved_[m] == 0 && prevMag_[m] > 0.0f)
                heap.push({prevMag_[m], m, 0});
        }

        const float binW = 2.0f * static_cast<float>(M_PI) / (2 * (bins_ - 1));
        while (unsolved > 0 && !heap.empty()) {
            const Entry e = heap.top();
            heap.pop();
            const int m = e.bin;
            if (e.frame == 0) {
                if (solved_[m] == 0) {
                    // time-direction: trapezoidal integration of omega
                    outPhase[m] = phase_[m] +
                                  0.5f * hs * (prevOmega_[m] + cur.omega[m]);
                    solved_[m] = 1;
                    --unsolved;
                    heap.push({cur.mag[m], m, 1});
                }
            } else {
                // frequency-direction from solved neighbour
                // dphi/domega = -alpha * tau  (group delay in output time)
                if (m + 1 < bins_ && solved_[m + 1] == 0) {
                    outPhase[m + 1] = outPhase[m] -
                        0.5f * binW * alpha * (cur.tau[m] + cur.tau[m + 1]);
                    solved_[m + 1] = 1;
                    --unsolved;
                    heap.push({cur.mag[m + 1], m + 1, 1});
                }
                if (m - 1 >= 0 && solved_[m - 1] == 0) {
                    outPhase[m - 1] = outPhase[m] +
                        0.5f * binW * alpha * (cur.tau[m] + cur.tau[m - 1]);
                    solved_[m - 1] = 1;
                    --unsolved;
                    heap.push({cur.mag[m - 1], m - 1, 1});
                }
            }
        }

        // leftover significant bins with no propagation path: keep continuity
        for (int m = 0; m < bins_; ++m)
            if (solved_[m] == 0)
                outPhase[m] = phase_[m] + hs * cur.omega[m];

        // below-tolerance bins: deterministic position-seeded random phase
        for (int m = 0; m < bins_; ++m)
            if (solved_[m] == 2)
                outPhase[m] = randomPhase(frameIndex, static_cast<uint64_t>(m));

        finish(cur, outPhase);
    }

    void setTolerance(float rel) { tolRel_ = rel; }

    // the last stepLocked() rebuilt at least one gliding-partial region
    bool chirpFound() const { return chirpFound_; }

    // Overwrite the stored synthesis phase of one bin (a caller that renders
    // a bin by other means keeps the next frame's propagation continuous).
    void setPhase(int m, float ph) {
        materialize();
        if (m >= 0 && m < bins_) phase_[m] = ph;
    }

    // One rigid rotation of the whole frame: every bin keeps its analysis
    // phase plus the time-propagated phase advance of the strongest peak.
    // The rendered grain is a phase-rotated copy of the windowed input, so
    // its envelope (and that of any normalised overlap-add of such grains)
    // stays within the input's; used where the frame holds no trustworthy
    // partial structure to lock (the edge of a stream's padding).
    void stepRigid(const AnalysisFrame& cur, int hs, std::vector<float>& outPhase) {
        outPhase.resize(bins_);
        const int B = bins_;
        if (!havePrev_) {
            for (int m = 0; m < B; ++m) outPhase[m] = std::arg(cur.X[m]);
            finish(cur, outPhase);
            return;
        }
        int p = -1;
        float best = 0.0f;
        for (int m = 1; m + 1 < B; ++m)
            if (cur.mag[m] > best && cur.mag[m] >= cur.mag[m - 1] &&
                cur.mag[m] > cur.mag[m + 1]) {
                best = cur.mag[m];
                p = m;
            }
        float delta = 0.0f;
        if (p > 0) {
            const double phSyn =
                prevPhase(p) + 0.5 * hs * (static_cast<double>(prevOmega_[p]) + cur.omega[p]);
            delta = princarg(phSyn - std::arg(cur.X[p]));
        }
        for (int m = 0; m < B; ++m) outPhase[m] = std::arg(cur.X[m]) + delta;
        finish(cur, outPhase);
    }

    // Record the frame that was actually rendered (its analysis and the
    // spectrum it was rendered with, by whatever path): the next step then
    // propagates from what was heard, not from a phase that was not used.
    // The phases are taken lazily, only where a later step reads them (the
    // peak bins): arg(Y[m]), or fallback[m] where Y[m] is exactly zero.
    void commit(const AnalysisFrame& cur, const std::vector<std::complex<float>>& Y,
                const std::vector<float>& fallback) {
        std::copy(Y.begin(), Y.begin() + bins_, lazyY_.begin());
        std::copy(fallback.begin(), fallback.begin() + bins_, lazyFallback_.begin());
        prevOmega_.assign(cur.omega.begin(), cur.omega.end());
        prevMag_.assign(cur.mag.begin(), cur.mag.end());
        havePrev_ = true;
        lazy_ = true;
    }

    // Marks (mask[m] = 1) the bins of every peak region of `cur` (the
    // stepLocked segmentation) whose peak bin is flagged in `peakOwn` (a
    // partial of this channel that the reference channel does not share).
    // Returns the number of such regions.
    int ownRegions(const AnalysisFrame& cur, const uint8_t* peakOwn,
                   std::vector<uint8_t>& mask) {
        const int B = bins_;
        mask.assign(B, 0);
        float maxMag = 0.0f;
        for (int m = 0; m < B; ++m)
            if (cur.mag[m] > maxMag) maxMag = cur.mag[m];
        const float tol = tolRel_ * maxMag;
        peaks_.clear();
        for (int m = 1; m + 1 < B; ++m)
            if (cur.mag[m] > tol && cur.mag[m] >= cur.mag[m - 1] &&
                cur.mag[m] > cur.mag[m + 1])
                peaks_.push_back(m);
        int lo = 0, n = 0;
        for (size_t k = 0; k < peaks_.size(); ++k) {
            const int p = peaks_[k];
            int hi = B;
            if (k + 1 < peaks_.size()) {
                int v = p;
                float vm = cur.mag[p];
                for (int m = p + 1; m <= peaks_[k + 1]; ++m)
                    if (cur.mag[m] < vm) { vm = cur.mag[m]; v = m; }
                hi = v + 1;
            }
            if (peakOwn[p]) {
                for (int m = lo; m < hi; ++m) mask[m] = 1;
                ++n;
            }
            lo = hi;
        }
        return n;
    }

    // Identity-phase-locked alternative (Laroche–Dolson with reassignment
    // omega): spectrum is segmented into peak regions at magnitude valleys;
    // each region is rotated rigidly so the analysis phase STRUCTURE
    // (mainlobe + sidelobes + inter-bin interference) is preserved exactly.
    // Cures the phase-inconsistency noise of smooth-phase integration on
    // tonal material. resetMask bins force region delta to 0 (attack reset).
    // jitterAmt (optional, 0..1 per bin): Damskagg-style noisiness-scaled
    // phase randomization, +- amt*pi, deterministic position-seeded.
    // chirpAlpha > 0 (the engine stretch, != 1): a region whose reassigned
    // group delay runs linearly across a broadened main lobe is a gliding
    // partial (sweep, fast FM). Copying its analysis phase verbatim keeps the
    // input glide rate inside every grain, so overlapping grains hold the
    // partial at different frequencies at the same output instant and beat
    // against each other (measured: -7 dB on the upper half of a log sweep
    // at -12 st; under expansion a 10 dB ripple at the hop rate on a linear
    // sweep at +21 st). Such a region is rebuilt instead from its
    // time-propagated peak phase and the group delay scaled by the stretch
    // (the stationary-phase solution of a time-scaled chirp: every frequency
    // of the glide lands at alpha * tau, so all grains agree on when the
    // partial passes each frequency). Stationary, decaying or attacking
    // partials (flat tau) are not affected. Under expansion only fast glides
    // qualify (see chirpExpand).
    // chirpGain (optional, size bins): receives the level factor of the
    // rebuilt glide regions, 1 elsewhere (see chirpLevelGain, chirpExpand).
    // onlyRegions (optional, size bins): phases (and gains) are computed only
    // for regions whose peak bin is flagged; the others are left undefined
    // (a caller that renders just those regions), while the per-region
    // glide history is still kept for every region.
    void stepLocked(const AnalysisFrame& cur, int hs,
                    const uint8_t* resetMask, std::vector<float>& outPhase,
                    const float* jitterAmt = nullptr,
                    uint64_t frameIndex = 0, float chirpAlpha = 0.0f,
                    float* chirpGain = nullptr,
                    const uint8_t* onlyRegions = nullptr) {
        outPhase.resize(bins_);
        const int B = bins_;
        if (chirpGain)
            for (int m = 0; m < B; ++m) chirpGain[m] = 1.0f;
        chirpFound_ = false;
        float maxMag = 0.0f;
        for (int m = 0; m < B; ++m)
            if (cur.mag[m] > maxMag) maxMag = cur.mag[m];
        const float tol = tolRel_ * maxMag;

        // fast-glide candidates of the previous frame (valid only if this was
        // called for the immediately preceding frame index), fresh ones for
        // this frame
        if (frameIndex != 0 && xCandFrame_ + 1 == frameIndex) xCand_.swap(xCandPrev_);
        else std::fill(xCandPrev_.begin(), xCandPrev_.end(), 0);
        std::fill(xCand_.begin(), xCand_.end(), 0);
        xCandFrame_ = frameIndex;

        if (!havePrev_) {
            for (int m = 0; m < B; ++m) outPhase[m] = std::arg(cur.X[m]);
            finish(cur, outPhase);
            return;
        }

        // find magnitude peaks
        peaks_.clear();
        for (int m = 1; m + 1 < B; ++m)
            if (cur.mag[m] > tol && cur.mag[m] >= cur.mag[m - 1] &&
                cur.mag[m] > cur.mag[m + 1])
                peaks_.push_back(m);

        if (peaks_.empty()) {
            for (int m = 0; m < B; ++m) outPhase[m] = std::arg(cur.X[m]);
            finish(cur, outPhase);
            return;
        }

        int lo = 0;  // region start
        for (size_t k = 0; k < peaks_.size(); ++k) {
            const int p = peaks_[k];
            int hi = B;  // region end (exclusive)
            if (k + 1 < peaks_.size()) {
                // valley between this peak and the next
                int v = p;
                float vm = cur.mag[p];
                for (int m = p + 1; m <= peaks_[k + 1]; ++m)
                    if (cur.mag[m] < vm) {
                        vm = cur.mag[m];
                        v = m;
                    }
                hi = v + 1;
            }
            if (onlyRegions && !onlyRegions[p]) {
                // not rendered by the caller: only the glide history
                if (chirpAlpha > 1.0f && !(resetMask && resetMask[p]) &&
                    isChirp(cur, lo, hi, p) && chirpFastGlide(cur, lo, hi))
                    for (int m = lo; m < hi; ++m) xCand_[m] = 1;
                lo = hi;
                continue;
            }
            const float phIn = std::arg(cur.X[p]);
            float delta;
            if (resetMask && resetMask[p]) {
                delta = 0.0f;  // synchronized attack reset: verbatim phases
            } else {
                // accumulate in double and wrap: keeps float precision over
                // arbitrarily long streams
                const double phSyn =
                    prevPhase(p) + 0.5 * hs * (static_cast<double>(prevOmega_[p]) +
                                               cur.omega[p]);
                delta = princarg(phSyn - phIn);
                if (chirpAlpha > 0.0f && isChirp(cur, lo, hi, p)) {
                    const float binW = 2.0f * static_cast<float>(M_PI) / (2 * (B - 1));
                    // expansion: per-bin gains, fast glides only, and only
                    // a glide the previous frame already held there (a
                    // sustained trajectory, not one frame's coincidence)
                    if (chirpAlpha > 1.0f) {
                        const bool fast = chirpFastGlide(cur, lo, hi);
                        if (fast)
                            for (int m = lo; m < hi; ++m) xCand_[m] = 1;
                        if (!fast || !xCandPrev_[p]) {
                            for (int m = lo; m < hi; ++m)
                                outPhase[m] = std::arg(cur.X[m]) + delta;
                            lo = hi;
                            continue;
                        }
                        if (chirpGain) chirpExpandGain(cur, lo, hi, chirpAlpha, chirpGain);
                    }
                    outPhase[p] = princarg(phSyn);
                    for (int m = p + 1; m < hi; ++m)
                        outPhase[m] = outPhase[m - 1] -
                            0.5f * binW * chirpAlpha * (cur.tau[m - 1] + cur.tau[m]);
                    for (int m = p - 1; m >= lo; --m)
                        outPhase[m] = outPhase[m + 1] +
                            0.5f * binW * chirpAlpha * (cur.tau[m] + cur.tau[m + 1]);
                    if (chirpGain && chirpAlpha < 1.0f) {
                        const float G = chirpLevelGain(cur, lo, hi, chirpAlpha);
                        for (int m = lo; m < hi; ++m) chirpGain[m] = G;
                    }
                    chirpFound_ = true;
                    lo = hi;
                    continue;
                }
            }
            for (int m = lo; m < hi; ++m)
                outPhase[m] = std::arg(cur.X[m]) + delta;
            lo = hi;
        }
        for (int m = lo; m < B; ++m) outPhase[m] = std::arg(cur.X[m]);
        if (jitterAmt) {
            for (int m = 0; m < B; ++m) {
                const float a = jitterAmt[m];
                if (a > 0.001f && !(resetMask && resetMask[m]))
                    outPhase[m] += a * (randomPhase(frameIndex, m) -
                                        static_cast<float>(M_PI));
            }
        }
        finish(cur, outPhase);
    }

    // Coherence-locked identity phase. stepLocked() takes the analysis phase
    // arg(X[m]) verbatim as each peak region's base, which copies the intra-
    // window group delay (the pulse-envelope position) unchanged. When the
    // synthesis hop exceeds the analysis hop (time-stretch), every overlapping
    // frame then re-deposits the SAME pulse at analysis-hop spacing — a hop-
    // synchronous comb, heard as a "fine DelayEcho". Here we instead anchor
    // each region at its peak's time-propagated synthesis phase and rebuild the
    // phase ACROSS the region from the reassignment group delay tau scaled by
    // alpha, so the synthesis group delay = alpha * analysis group delay: the
    // pulse envelope is relocated onto the stretched grid and overlapping
    // frames reinforce instead of combing (spectrum-wide phase coherence, the
    // property that keeps a phase-locked vocoder from sounding phasy/echoey).
    // Reuses the exact alpha*tau frequency propagation from step().
    void stepLockedCoherent(const AnalysisFrame& cur, int hs, float alpha,
                            const uint8_t* resetMask,
                            std::vector<float>& outPhase, bool rawPhase = false,
                            bool suppress = false) {
        materialize();
        outPhase.resize(bins_);
        const int B = bins_;
        float maxMag = 0.0f;
        for (int m = 0; m < B; ++m)
            if (cur.mag[m] > maxMag) maxMag = cur.mag[m];
        const float tol = tolRel_ * maxMag;

        if (!havePrev_) {
            for (int m = 0; m < B; ++m) outPhase[m] = std::arg(cur.X[m]);
            finish(cur, outPhase);
            return;
        }

        peaks_.clear();
        for (int m = 1; m + 1 < B; ++m)
            if (cur.mag[m] > tol && cur.mag[m] >= cur.mag[m - 1] &&
                cur.mag[m] > cur.mag[m + 1])
                peaks_.push_back(m);

        if (peaks_.empty()) {
            for (int m = 0; m < B; ++m) outPhase[m] = std::arg(cur.X[m]);
            finish(cur, outPhase);
            return;
        }

        // Energy-rising transient-partial suppression. A broadband hit (snare)
        // spikes many bins into separate peaks; each would be phase-locked as an
        // INDEPENDENT partial, and the tiny per-partial phase differences read as
        // a fan of detuned copies = attack "delay"/chorus. Merge every interior
        // partial whose energy jumped > k x the previous frame's energy at that
        // bin into the neighbouring region (drop it as a lock centre), keeping the
        // single strongest peak and the two spectral edges, so few coherent
        // centres govern the whole click = one sharp impulse.
        if (suppress && havePrev_ && peaks_.size() > 2) {
            int strongest = peaks_[0];
            for (int p : peaks_)
                if (cur.mag[p] > cur.mag[strongest]) strongest = p;
            const float k = 1.5f;
            supPeaks_.clear();
            for (size_t i = 0; i < peaks_.size(); ++i) {
                const int p = peaks_[i];
                const bool edge = (i == 0 || i + 1 == peaks_.size());
                const bool rising = cur.mag[p] > k * prevMag_[p];
                if (p == strongest || edge || !rising)
                    supPeaks_.push_back(p);  // keep; rising interior peaks drop
            }
            peaks_.swap(supPeaks_);
        }

        const float binW = 2.0f * static_cast<float>(M_PI) / (2 * (B - 1));
        int lo = 0;  // region start (inclusive)
        for (size_t k = 0; k < peaks_.size(); ++k) {
            const int p = peaks_[k];
            int hi = B;  // region end (exclusive)
            if (k + 1 < peaks_.size()) {
                int v = p;
                float vm = cur.mag[p];
                for (int m = p + 1; m <= peaks_[k + 1]; ++m)
                    if (cur.mag[m] < vm) { vm = cur.mag[m]; v = m; }
                hi = v + 1;
            }
            // peak's absolute synthesis phase: verbatim on attack reset, else
            // the time-propagated (trapezoidal omega) running phase.
            if (resetMask && resetMask[p]) {
                outPhase[p] = std::arg(cur.X[p]);
            } else {
                const double phSyn =
                    phase_[p] + 0.5 * hs * (static_cast<double>(prevOmega_[p]) +
                                            cur.omega[p]);
                outPhase[p] = princarg(phSyn);
            }
            // rebuild the region's phase, propagating outward from the peak.
            if (rawPhase) {
                // scale the RAW inter-bin phase difference by alpha (mirrors a
                // shape-invariant frequency-direction integration). Raw phase is
                // ALWAYS defined, so it relocates the intra-window group delay of
                // a BROADBAND transient (snare) onto the stretched grid without
                // the reassignment-tau smear that tau-integration leaves on
                // non-tonal hits -> sharp, chorus-free percussive attacks.
                for (int m = p + 1; m < hi; ++m)
                    outPhase[m] = outPhase[m - 1] +
                        alpha * princarg(std::arg(cur.X[m]) - std::arg(cur.X[m - 1]));
                for (int m = p - 1; m >= lo; --m)
                    outPhase[m] = outPhase[m + 1] -
                        alpha * princarg(std::arg(cur.X[m + 1]) - std::arg(cur.X[m]));
            } else {
                // tonal path: (alpha-scaled) reassignment group delay.
                for (int m = p + 1; m < hi; ++m)
                    outPhase[m] = outPhase[m - 1] -
                        0.5f * binW * alpha * (cur.tau[m - 1] + cur.tau[m]);
                for (int m = p - 1; m >= lo; --m)
                    outPhase[m] = outPhase[m + 1] +
                        0.5f * binW * alpha * (cur.tau[m] + cur.tau[m + 1]);
            }
            lo = hi;
        }
        for (int m = lo; m < B; ++m) outPhase[m] = std::arg(cur.X[m]);
        finish(cur, outPhase);
    }

    // Shape-invariant voice phase (SHIP-style harmonic locking). For voiced
    // frames with a confident F0, the glottal pulse shape is defined by the
    // phase RELATIONSHIP between harmonics: phi_h - h*phi_1 must be preserved.
    // Standard peak-region locking propagates each harmonic independently, so
    // small per-harmonic errors accumulate and smear the pulse ("phasy" /
    // "buzzy" voice). Here we propagate only the fundamental's phase and set
    // every harmonic's synthesis phase to h*phi_1_syn + (input harmonic-to-
    // fundamental offset), which keeps the pulse shape intact while advancing
    // the pulse train in time. Falls back to identity-lock if F0 is unreliable.
    // f0Bin: fractional bin index of the fundamental (<=0 = unvoiced -> caller
    // should use stepLocked instead).
    void stepVoice(const AnalysisFrame& cur, int hs, double f0Bin,
                   std::vector<float>& outPhase) {
        materialize();
        outPhase.resize(bins_);
        const int B = bins_;
        float maxMag = 0.0f;
        for (int m = 0; m < B; ++m)
            if (cur.mag[m] > maxMag) maxMag = cur.mag[m];
        const float tol = tolRel_ * maxMag;
        if (!havePrev_) {
            for (int m = 0; m < B; ++m) outPhase[m] = std::arg(cur.X[m]);
            finish(cur, outPhase);
            return;
        }

        // locate the fundamental peak: strongest peak within +-15% of f0Bin
        int p1 = -1;
        float best = 0.0f;
        const int lo0 = std::max(1, static_cast<int>(f0Bin * 0.85));
        const int hi0 = std::min(B - 2, static_cast<int>(f0Bin * 1.15 + 1));
        for (int m = lo0; m <= hi0; ++m)
            if (cur.mag[m] > best && cur.mag[m] >= cur.mag[m - 1] &&
                cur.mag[m] >= cur.mag[m + 1]) {
                best = cur.mag[m];
                p1 = m;
            }
        if (p1 < 1 || best < tol) {  // no clear fundamental -> identity lock
            stepLocked(cur, hs, nullptr, outPhase);
            return;
        }

        // propagate the fundamental phase (trapezoidal, double-accumulated)
        const double phi1In = std::arg(cur.X[p1]);
        const double phi1Syn =
            phase_[p1] + 0.5 * hs * (static_cast<double>(prevOmega_[p1]) +
                                     cur.omega[p1]);

        // all magnitude peaks -> region deltas
        peaks_.clear();
        for (int m = 1; m + 1 < B; ++m)
            if (cur.mag[m] > tol && cur.mag[m] >= cur.mag[m - 1] &&
                cur.mag[m] > cur.mag[m + 1])
                peaks_.push_back(m);
        if (peaks_.empty()) {
            stepLocked(cur, hs, nullptr, outPhase);
            return;
        }

        int lo = 0;
        for (size_t k = 0; k < peaks_.size(); ++k) {
            const int p = peaks_[k];
            int hi = B;
            if (k + 1 < peaks_.size()) {
                int v = p;
                float vm = cur.mag[p];
                for (int m = p + 1; m <= peaks_[k + 1]; ++m)
                    if (cur.mag[m] < vm) { vm = cur.mag[m]; v = m; }
                hi = v + 1;
            }
            const double phIn = std::arg(cur.X[p]);
            // harmonic number of this peak relative to the fundamental
            const double hNum = static_cast<double>(p) / p1;
            double delta;
            if (hNum >= 0.5 && hNum < 10.5) {
                // Lock only the low harmonics that define the glottal pulse
                // shape. High harmonics take ordinary propagation: h*phi1
                // wraps many times for large h and the offset loses precision,
                // which measured WORSE at +24 st and on dense low-F0 frames.
                const int h = static_cast<int>(hNum + 0.5);
                // shape offset from the input; synthesis = h*phi1Syn + offset
                const double off = princarg(phIn - h * phi1In);
                const double phSyn = h * phi1Syn + off;
                delta = princarg(phSyn - phIn);
            } else if (hNum >= 10.5) {
                const double phSyn =
                    phase_[p] + 0.5 * hs *
                                    (static_cast<double>(prevOmega_[p]) +
                                     cur.omega[p]);
                delta = princarg(phSyn - phIn);
            } else {
                // sub-fundamental content: ordinary propagation
                const double phSyn =
                    phase_[p] + 0.5 * hs *
                                    (static_cast<double>(prevOmega_[p]) +
                                     cur.omega[p]);
                delta = princarg(phSyn - phIn);
            }
            for (int m = lo; m < hi; ++m)
                outPhase[m] = std::arg(cur.X[m]) + static_cast<float>(delta);
            lo = hi;
        }
        for (int m = lo; m < B; ++m) outPhase[m] = std::arg(cur.X[m]);
        finish(cur, outPhase);
    }

    // Level factor of a glide region rebuilt with alpha-scaled group delay
    // (compression). The rebuild squeezes the grain's content in time with
    // the glide (to alpha of the window for a fast glide) while the overlap-
    // add norm assumes it spans the analysis window, so the level drops
    // (alpha 0.5: -1 dB, alpha 0.25: -3.4 dB, measured on a log sweep). For a
    // linear chirp under a Gaussian window of width s (the Hann main lobe has
    // the same spread), with B = chirp rate * s^2: the spectrum spreads by
    // (1 + B^2), the rebuilt grain's envelope has width q*s,
    // q^2 = (1 + a^2 B^2)/(1 + B^2), its peak is sqrt(1/q) (energy is
    // preserved), and the normalised overlap-add of such grains gives
    // sqrt(2q/(1 + q^2)). B^2 comes from the region's measured spectral
    // spread against a stationary Hann partial's (1/3 bin^2 over its main
    // lobe). Returns the inverse of that level (the ideal output frame would
    // span more bins than the input frame holds, so the region is raised as
    // a whole rather than reshaped).
    float chirpLevelGain(const AnalysisFrame& cur, int lo, int hi, float a) const {
        double sw = 0.0, sm = 0.0;
        for (int m = lo; m < hi; ++m) {
            const double w = static_cast<double>(cur.mag[m]) * cur.mag[m];
            sw += w;
            sm += w * m;
        }
        if (sw <= 0.0) return 1.0f;
        const double mc = sm / sw;
        double var = 0.0;
        for (int m = lo; m < hi; ++m)
            var += static_cast<double>(cur.mag[m]) * cur.mag[m] * (m - mc) * (m - mc);
        var /= sw;
        const double B2 = std::max(0.0, 3.0 * var - 1.0);
        const double q = std::sqrt((1.0 + static_cast<double>(a) * a * B2) / (1.0 + B2));
        const double L = std::sqrt(2.0 * q / (1.0 + q * q));
        return static_cast<float>(std::clamp(1.0 / L, 1.0, 2.0));
    }

    // Glide region under EXPANSION (a > 1). The alpha-scaled group delay
    // relocates each bin's share of the glide from tau to a*tau. By
    // stationary phase a bin with reassigned delay tau carries the part of the
    // glide that passed the window centre tau samples away, |X| ~ w(tau) (w =
    // the analysis Hann), so the relocated grain alone would have the envelope
    // w(t / a) / sqrt(a): wider than the window (the circular synthesis wraps
    // whatever lies beyond N/2 onto the grain's other side) and, normalised
    // by sum w*w, too weak where it is kept. Scaling each bin by
    //     sqrt(a) * w(a * tau) / w(tau)
    // turns the grain into w(t) times the time-scaled glide (rate / a), which
    // the WOLA reconstructs exactly: every grain holds the glide where the
    // time map puts it, with the analysis window's own envelope (measured:
    // the 10 dB hop-rate ripple of a linear sweep at +14/+21 st and 2x
    // stretch flattened to +-0.1 dB, at the input level). Bins relocated past
    // the window are left to the grains centred on them. This stationary-
    // phase solution needs a glide that sweeps well beyond the main lobe in
    // one window; with B^2 (from the region's spectral spread, as in
    // chirpLevelGain) below 10 the spread is not dominated by the glide (a
    // moderate speech glide carries as much amplitude change and curvature),
    // the model does not hold (the Gaussian-window solution for that regime
    // measured 0.25-0.45 dB low on sweeps and lowered speech NISQA at 2-4x
    // stretch) and the region keeps the identity lock. A glide must also have
    // been there in the previous frame (a sustained trajectory; an isolated
    // frame of a speech glide or formant move is left alone).
    bool chirpFastGlide(const AnalysisFrame& cur, int lo, int hi) const {
        double sw = 0.0, sm = 0.0;
        for (int m = lo; m < hi; ++m) {
            const double w = static_cast<double>(cur.mag[m]) * cur.mag[m];
            sw += w;
            sm += w * m;
        }
        if (sw <= 0.0) return false;
        const double mc = sm / sw;
        double var = 0.0;
        for (int m = lo; m < hi; ++m)
            var += static_cast<double>(cur.mag[m]) * cur.mag[m] * (m - mc) * (m - mc);
        var /= sw;
        const double B2 = std::max(0.0, 3.0 * var - 1.0);
        return B2 >= 10.0;
    }
    void chirpExpandGain(const AnalysisFrame& cur, int lo, int hi, float a,
                         float* gain) const {
        const double n = 2.0 * (bins_ - 1);
        auto hann = [&](double t) {
            if (std::abs(t) >= 0.5 * n) return 0.0;
            return 0.5 + 0.5 * std::cos(2.0 * M_PI * t / n);
        };
        const double sa = std::sqrt(static_cast<double>(a));
        for (int m = lo; m < hi; ++m) {
            const double w0 = hann(cur.tau[m]);
            gain[m] = w0 > 1e-6 ? static_cast<float>(sa * hann(a * cur.tau[m]) / w0) : 0.0f;
        }
    }

    // Marks (mask[m] = 1) the bins of every peak region of `cur` that
    // isChirp() classifies as a gliding partial, with the same peak / valley
    // segmentation stepLocked() uses. Returns the number of such regions.
    int chirpRegions(const AnalysisFrame& cur, std::vector<uint8_t>& mask) {
        const int B = bins_;
        mask.assign(B, 0);
        float maxMag = 0.0f;
        for (int m = 0; m < B; ++m)
            if (cur.mag[m] > maxMag) maxMag = cur.mag[m];
        const float tol = tolRel_ * maxMag;
        peaks_.clear();
        for (int m = 1; m + 1 < B; ++m)
            if (cur.mag[m] > tol && cur.mag[m] >= cur.mag[m - 1] &&
                cur.mag[m] > cur.mag[m + 1])
                peaks_.push_back(m);
        int lo = 0, n = 0;
        for (size_t k = 0; k < peaks_.size(); ++k) {
            const int p = peaks_[k];
            int hi = B;
            if (k + 1 < peaks_.size()) {
                int v = p;
                float vm = cur.mag[p];
                for (int m = p + 1; m <= peaks_[k + 1]; ++m)
                    if (cur.mag[m] < vm) { vm = cur.mag[m]; v = m; }
                hi = v + 1;
            }
            if (isChirp(cur, lo, hi, p)) {
                for (int m = lo; m < hi; ++m) mask[m] = 1;
                ++n;
            }
            lo = hi;
        }
        return n;
    }

    // Gliding-partial test for one peak region [lo, hi): an isolated partial
    // whose bins within 20 dB of the peak (at least 5: a stationary Hann
    // main lobe has 3-4) carry a reassigned group delay that is linear in
    // frequency (magnitude^2-weighted fit, R^2 >= 0.9) and spans at least a
    // quarter of the window, i.e. the partial's frequency moves by several
    // bins within one analysis window. Stationary partials have tau ~ 0 and
    // amplitude-modulated ones a flat tau offset, so neither qualifies.
    bool isChirp(const AnalysisFrame& cur, int lo, int hi, int p) const {
        const float thr = 0.1f * cur.mag[p];
        // an isolated partial: both region edges (the valleys towards the
        // neighbouring peaks) lie 20 dB under it. Glides wider than their
        // neighbour spacing (the upper harmonics of a fast speech glide)
        // smear into each other; their region is not one partial.
        if (cur.mag[lo] >= thr || cur.mag[hi - 1] >= thr) return false;
        double sw = 0.0, sx = 0.0, sy = 0.0;
        int cnt = 0, mlo = p, mhi = p;
        for (int m = lo; m < hi; ++m) {
            if (cur.mag[m] < thr) continue;
            const double w = static_cast<double>(cur.mag[m]) * cur.mag[m];
            sw += w;
            sx += w * (m - p);
            sy += w * cur.tau[m];
            ++cnt;
            mlo = std::min(mlo, m);
            mhi = std::max(mhi, m);
        }
        if (cnt < 5 || sw <= 0.0) return false;
        const double mx = sx / sw, my = sy / sw;
        double sxx = 0.0, sxy = 0.0, syy = 0.0;
        for (int m = lo; m < hi; ++m) {
            if (cur.mag[m] < thr) continue;
            const double w = static_cast<double>(cur.mag[m]) * cur.mag[m];
            const double dx = (m - p) - mx, dy = cur.tau[m] - my;
            sxx += w * dx * dx;
            sxy += w * dx * dy;
            syy += w * dy * dy;
        }
        if (sxx <= 0.0 || syy <= 0.0) return false;
        const double slope = sxy / sxx;
        const double r2 = sxy * sxy / (sxx * syy);
        const double span = std::abs(slope) * (mhi - mlo);
        const double n = 2.0 * (bins_ - 1);
        return r2 >= 0.9 && span >= 0.25 * n;
    }

private:
    void finish(const AnalysisFrame& cur, const std::vector<float>& ph) {
        phase_ = ph;
        prevOmega_.assign(cur.omega.begin(), cur.omega.end());
        prevMag_.assign(cur.mag.begin(), cur.mag.end());
        havePrev_ = true;
        lazy_ = false;
    }

    // previous synthesis phase of bin m (see commit)
    float prevPhase(int m) const {
        if (!lazy_) return phase_[m];
        return lazyY_[m] != std::complex<float>(0.0f, 0.0f) ? std::arg(lazyY_[m])
                                                            : lazyFallback_[m];
    }
    // all of them, for the steps that read every bin's previous phase
    void materialize() {
        if (!lazy_) return;
        for (int m = 0; m < bins_; ++m) phase_[m] = prevPhase(m);
        lazy_ = false;
    }

    static float princarg(double x) {
        const double twoPi = 2.0 * M_PI;
        x -= twoPi * std::floor(x / twoPi + 0.5);
        return static_cast<float>(x);
    }

    static float randomPhase(uint64_t frame, uint64_t bin) {
        // splitmix64 on (frame, bin) -> [0, 2pi); fully deterministic
        uint64_t z = frame * 0x9E3779B97F4A7C15ull + bin + 0x632BE59BD9B4E019ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return static_cast<float>(
            (z >> 11) * (1.0 / 9007199254740992.0) * 2.0 * M_PI);
    }

    int bins_;
    float tolRel_ = 1e-3f;  // magnitude-domain (1e-6 energy)
    bool havePrev_;
    bool chirpFound_ = false;
    std::vector<float> phase_, prevOmega_, prevMag_;
    std::vector<uint8_t> xCand_, xCandPrev_;  // fast-glide bins (this / last frame)
    uint64_t xCandFrame_ = 0;                 // frame index xCand_ was filled for
    std::vector<std::complex<float>> lazyY_;  // committed spectrum (see commit)
    std::vector<float> lazyFallback_;
    bool lazy_ = false;
    std::vector<uint8_t> solved_;
    std::vector<int> peaks_;
    std::vector<int> supPeaks_;  // scratch for partial suppression
};

}  // namespace pbshift
