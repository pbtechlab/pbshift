// Real-Time Phase Gradient Heap Integration (RTPGHI).
// Prusa & Sondergaard 2016/2017, implemented cleanroom from the papers.
//
// Given the previous synthesis frame's phases and both frames' measured
// phase gradients, integrates phases across the current frame: significant
// bins get heap-ordered trapezoidal integration (time direction from the
// previous frame, frequency direction from already-solved neighbours);
// bins below tolerance get deterministic position-seeded random phase.
#pragma once
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
        solved_.resize(bins);
        havePrev_ = false;
    }

    void reset() {
        havePrev_ = false;
        std::fill(phase_.begin(), phase_.end(), 0.0f);
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

    // Identity-phase-locked alternative (Laroche–Dolson with reassignment
    // omega): spectrum is segmented into peak regions at magnitude valleys;
    // each region is rotated rigidly so the analysis phase STRUCTURE
    // (mainlobe + sidelobes + inter-bin interference) is preserved exactly.
    // Cures the phase-inconsistency noise of smooth-phase integration on
    // tonal material. resetMask bins force region delta to 0 (attack reset).
    // jitterAmt (optional, 0..1 per bin): Damskagg-style noisiness-scaled
    // phase randomization, +- amt*pi, deterministic position-seeded.
    void stepLocked(const AnalysisFrame& cur, int hs,
                    const uint8_t* resetMask, std::vector<float>& outPhase,
                    const float* jitterAmt = nullptr,
                    uint64_t frameIndex = 0) {
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
            const float phIn = std::arg(cur.X[p]);
            float delta;
            if (resetMask && resetMask[p]) {
                delta = 0.0f;  // synchronized attack reset: verbatim phases
            } else {
                // accumulate in double and wrap: keeps float precision over
                // arbitrarily long streams
                const double phSyn =
                    phase_[p] + 0.5 * hs * (static_cast<double>(prevOmega_[p]) +
                                            cur.omega[p]);
                delta = princarg(phSyn - phIn);
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
                            std::vector<float>& outPhase) {
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
            // rebuild the region's phase from the (alpha-scaled) group delay,
            // propagating outward from the peak in both directions.
            for (int m = p + 1; m < hi; ++m)
                outPhase[m] = outPhase[m - 1] -
                    0.5f * binW * alpha * (cur.tau[m - 1] + cur.tau[m]);
            for (int m = p - 1; m >= lo; --m)
                outPhase[m] = outPhase[m + 1] +
                    0.5f * binW * alpha * (cur.tau[m] + cur.tau[m + 1]);
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

private:
    void finish(const AnalysisFrame& cur, const std::vector<float>& ph) {
        phase_ = ph;
        prevOmega_.assign(cur.omega.begin(), cur.omega.end());
        prevMag_.assign(cur.mag.begin(), cur.mag.end());
        havePrev_ = true;
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
    std::vector<float> phase_, prevOmega_, prevMag_;
    std::vector<uint8_t> solved_;
    std::vector<int> peaks_;
};

}  // namespace pbshift
