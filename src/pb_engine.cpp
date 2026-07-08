// pbshift engine core: streaming scheduler around the shared analysis
// front-end, RTPGHI phase engine and WOLA synthesis.
//
// M2 scope: time-stretch via RTPGHI with stereo reference-channel phase
// locking (inter-channel phase deltas copied verbatim from the input).
// Pitch shifting is done by the CLI via the front resampler (M3) until the
// resampler moves inside the engine.
#include "pbshift/pbshift.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "pb_envelope.h"
#include "pb_fft.h"
#include "pb_multires.h"
#include "pb_pghi.h"
#include "pb_resampler.h"
#include "pb_stft.h"
#include "pb_window.h"

namespace pbshift {

namespace {
int nextPow2(int v) {
    int n = 32;
    while (n < v) n <<= 1;
    return n;
}
}  // namespace

struct Stretcher::Impl {
    Config cfg;
    int N = 4096;      // FFT / window size
    int hs = 1024;     // synthesis hop (output samples)
    double alpha = 1.0;      // engine (WOLA) stretch = alphaUser * pitchFactor
    double alphaUser = 1.0;  // user-facing time stretch
    double pitchSemi = 0.0;
    bool formant = false;

    // pitch resampler: pre-stage for pitch down (slows articulation into the
    // engine), post-stage for pitch up (avoids compressing articulation)
    std::unique_ptr<SincResampler> resIn, resOut;
    std::vector<std::vector<float>> resBuf;
    std::vector<float*> resPtr;
    std::vector<const float*> cPtr;

    // ---- offline multi-resolution path (opt-in) -------------------------
    // When enabled (Offline tier + PBSHIFT_MULTIRES), the whole input is
    // buffered and time-stretched by MultiResStretch on finish(): a long
    // window for the low band (frequency resolution, kills chorusing) and a
    // short window for the high band (transient time resolution). Pitch is
    // still handled by the pre/post resamplers exactly as in the streaming
    // path, so this only replaces the "stretch" middle stage. Default builds
    // are untouched (useMultiRes stays false).
    bool useMultiRes = false;
    std::vector<std::vector<float>> mrIn;   // buffered (pre-resampled) input
    std::vector<std::vector<float>> mrOut;  // stretched (post-resampled) output
    long long mrReadPos = 0;
    bool mrDone = false;
    // Formant preservation is not implemented in the multires path; when it is
    // requested, fall back to the streaming engine (which has it). Evaluated
    // lazily because setFormantPreserve() is called after configure().
    bool mrActive() const { return useMultiRes && !formant; }

    double pitchFactor() const { return std::pow(2.0, pitchSemi / 12.0); }
    // Deep pitch-down splits the resampling across pre and post stages
    // (sqrt(p) each): a single x4 pre-resample squeezes harmonic spacing
    // below bin resolution (27.5 Hz fundamental = 2.3 bins at N=4096) and
    // the -24 st corner collapses (measured HNR 17 dB). Splitting keeps the
    // engine's view at sqrt(p) compression only.
    bool splitDown() const { return pitchSemi < -12.0; }
    double preRatio() const {  // input resampler ratio (>1 lengthens)
        if (pitchSemi >= 0.0) return 1.0;
        const double inv = 1.0 / pitchFactor();
        return splitDown() ? std::sqrt(inv) : inv;
    }
    double postRatio() const {  // output resampler ratio
        if (pitchSemi > 0.0) return 1.0 / pitchFactor();
        return splitDown() ? std::sqrt(1.0 / pitchFactor()) : 1.0;
    }
    void updateAlpha() {
        alpha = std::clamp(alphaUser * pitchFactor(), 0.02, 64.0);
        // deep compression: keep the ANALYSIS hop <= N/2 or magnitude
        // evolution gets temporally aliased (content skipped between frames)
        if (synthCount == 0) {
            int h = N / 4;
            while (alpha < 0.5 && h / alpha > N / 2 && h > N / 16) h /= 2;
            hs = h;
        }
    }

    // input ring (per channel), absolute indexing
    std::vector<std::vector<float>> inRing;
    int inCap = 0;
    long long totalIn = 0;
    bool finished = false;
    long long padAppended = 0;

    // analysis
    WindowSet win;
    std::vector<std::unique_ptr<AnalysisFrontend>> fe;
    std::vector<AnalysisFrame> frames;
    std::unique_ptr<Rtpghi> pghi;
    std::vector<float> synthPhase;

    // synthesis
    std::vector<std::unique_ptr<WolaSynth>> wola;
    long long synthCount = 0;   // frames synthesized
    double inPos = 0.0;         // input center of next analysis frame
    long long readPos = 0;      // absolute output read position
    int refCh = 0;

    std::vector<float> frameBuf;
    std::vector<std::complex<float>> Y;

    // ---- staged frame pipeline (split computation) -------------------
    // One frame's synthesis is split into 3*C+1 stages: analyze per
    // channel, formant per channel, ONE atomic transient/phase/advance
    // decision, render (Y + iFFT + WOLA) per channel. Each stage has a
    // due point that is a pure function of totalIn (never call boundaries
    // or wall time), so worst-case per-feed() cost approaches the average
    // while the output stays bit-identical for any input chunking.
    bool fActive = false;      // a frame is mid-flight
    int fStage = 0;            // next stage to execute
    long long fCIn = 0;        // frame center (input samples)
    long long fDueBase = 0;    // fCIn + N/2 = input point the frame starts
    long long fSpread = 0;     // due points spread over this input span
    double fBestE = -1.0;      // reference-channel election accumulator
    int fBest = 0;
    bool fDoFormant = false;   // per-frame snapshot of the formant path
    double fP = 1.0;           // pitch factor snapshot for formant gains
    int fEffOrder = 0;         // TrueEnvelope order snapshot
    float fFmtStrength = 1.0f; // formant-correction strength snapshot
    double fAdvance = 0.0;     // input advance decided by the atomic stage
    long long fCOut = 0;       // output center for the render stages

    // transient engine (M4)
    std::vector<float> prevRefMag;   // previous analysis frame magnitudes (ref ch)
    std::vector<std::vector<float>> prevMagCh;  // per-channel prev magnitudes
    std::vector<uint8_t> resetMask;
    float prevFlux = 0.0f;
    double debt = 0.0;               // input samples consumed ahead of schedule
    double lastOnsetIn = -1e18;      // input position of last accepted onset
    int pendCount = 0;               // frames spent holding a coming attack
    int pinRemain = 0;               // remaining ratio-1 pinned frames
    float tauMean = 0.0f;            // mag^2-weighted rise-bin group delay
    float lastFlux = 0.0f;           // detector observables of current frame
    float lastRiseFrac = 0.0f;

    // noisiness estimator (M5a, Damskagg-style fuzzy classification)
    static constexpr int kHist = 5;
    std::vector<std::vector<float>> magHist;  // ring of last kHist ref-mags
    int histCount = 0;
    std::vector<float> jitterAmt;

    // formant engine (M6)
    std::vector<std::unique_ptr<TrueEnvelope>> tenv;
    std::vector<float> envLog, fmtGain;
    std::vector<std::vector<float>> envSmooth;  // per-channel EMA state
    // Formant-gain temporal smoothing. The per-frame envelope flutters, which
    // amplitude-modulates the output = the "buzzy" phase-vocoder voice quality
    // (measured: +12 st voice 30-250 Hz modulation 22 dB with no smoothing vs
    // ~13 dB at 0.8). Default ON; the envelope-LSD
    // metric that argued against it is unreliable (it missed the pitch-cancel
    // bug too). Smooths the 30-250 Hz flutter while syllable-rate (<10 Hz)
    // formant motion passes.
    float envLambda = 0.85f;
    int teOrder = 256;       // base cepstral order (sweep-optimal at N=4096)

    void configure(const Config& c) {
        cfg = c;
        // window length by tier (round-trip latency = 1.5 N):
        //   Live      ~2048 @48k -> ~64 ms  (meets the <100 ms insert spec)
        //   StudioRT  ~4096 @48k -> ~128 ms (default, max real-time quality)
        //   Offline   ~8192 @48k -> ~256 ms (best bass resolution)
        double winSec = 0.085;  // -> 4096 @48k
        if (cfg.tier == Config::Tier::Live) winSec = 0.040;      // -> 2048
        else if (cfg.tier == Config::Tier::Offline) winSec = 0.17;  // -> 8192
        useMultiRes = cfg.tier == Config::Tier::Offline &&
                      std::getenv("PBSHIFT_MULTIRES") != nullptr;
        N = nextPow2(static_cast<int>(cfg.sampleRate * winSec));
        if (const char* f = std::getenv("PBSHIFT_FFT")) N = std::atoi(f);
        hs = N / 4;
        if (const char* h = std::getenv("PBSHIFT_HOPDIV")) hs = N / std::atoi(h);
        win = WindowSet::hann(N);
        inCap = 1 << 19;
        while (inCap < 8 * N) inCap <<= 1;
        inRing.assign(cfg.channels, std::vector<float>(inCap, 0.0f));
        fe.clear();
        wola.clear();
        frames.resize(cfg.channels);
        for (int ch = 0; ch < cfg.channels; ++ch) {
            fe.emplace_back(new AnalysisFrontend(N, win));
            wola.emplace_back(new WolaSynth(N, win, inCap));
        }
        pghi.reset(new Rtpghi(N / 2 + 1));
        if (const char* t = std::getenv("PBSHIFT_TOL"))
            pghi->setTolerance(static_cast<float>(std::atof(t)));
        teOrder = 256 * N / 4096;
        if (const char* o = std::getenv("PBSHIFT_TE_ORDER")) teOrder = std::atoi(o);
        if (const char* l = std::getenv("PBSHIFT_TE_SMOOTH"))
            envLambda = static_cast<float>(std::atof(l));
        int teSub = 0;
        if (const char* s = std::getenv("PBSHIFT_TE_SUB")) teSub = std::atoi(s);
        tenv.clear();
        for (int ch = 0; ch < cfg.channels; ++ch)
            tenv.emplace_back(new TrueEnvelope(N, teOrder, teSub, cfg.sampleRate));
        envSmooth.assign(cfg.channels, {});
        Y.resize(N / 2 + 1);
        frameBuf.resize(N);
        resetState();
    }

    void resetState() {
        for (auto& r : inRing) std::fill(r.begin(), r.end(), 0.0f);
        for (auto& w : wola) w->clear();
        pghi->reset();
        totalIn = 0;
        finished = false;
        padAppended = 0;
        synthCount = 0;
        inPos = 0.0;
        readPos = 0;
        refCh = 0;
        prevRefMag.clear();
        prevMagCh.assign(cfg.channels, {});
        resetMask.assign(N / 2 + 1, 0);
        prevFlux = 0.0f;
        debt = 0.0;
        lastOnsetIn = -1e18;
        pendCount = 0;
        pinRemain = 0;
        magHist.assign(kHist, std::vector<float>(N / 2 + 1, 0.0f));
        histCount = 0;
        jitterAmt.assign(N / 2 + 1, 0.0f);
        fActive = false;
        fStage = 0;
        fCIn = fDueBase = fSpread = 0;
        fBestE = -1.0;
        fBest = 0;
        fDoFormant = false;
        fP = 1.0;
        fEffOrder = 0;
        fAdvance = 0.0;
        fCOut = 0;
        resIn.reset(new SincResampler(cfg.channels, 64, 12.0));
        resOut.reset(new SincResampler(cfg.channels, 64, 12.0));
        resBuf.assign(cfg.channels, std::vector<float>(8192));
        resPtr.resize(cfg.channels);
        cPtr.resize(cfg.channels);
        for (int c = 0; c < cfg.channels; ++c) resPtr[c] = resBuf[c].data();
        engineHanded = 0;
        mrIn.assign(cfg.channels, {});
        mrOut.assign(cfg.channels, {});
        mrReadPos = 0;
        mrDone = false;
    }

    // ---- pitch-aware streaming wrappers -----------------------------
    void feedUser(const float* const* in, int nf) {
        if (preRatio() != 1.0) {
            // pre-resample (ratio > 1: more samples, lower pitch)
            resIn->feed(in, nf);
            const double ratio = preRatio();
            int made;
            do {
                made = resIn->process(resPtr.data(),
                                      static_cast<int>(resBuf[0].size()), ratio);
                if (made > 0) {
                    for (int c = 0; c < cfg.channels; ++c) cPtr[c] = resPtr[c];
                    feed(const_cast<const float* const*>(cPtr.data()), made);
                }
            } while (made == static_cast<int>(resBuf[0].size()));
        } else {
            feed(in, nf);
        }
    }

    void finishUser() {
        if (preRatio() != 1.0) {
            resIn->finish();
            const double ratio = preRatio();
            int made;
            do {
                made = resIn->process(resPtr.data(),
                                      static_cast<int>(resBuf[0].size()), ratio);
                if (made > 0) {
                    for (int c = 0; c < cfg.channels; ++c) cPtr[c] = resPtr[c];
                    feed(const_cast<const float* const*>(cPtr.data()), made);
                }
            } while (made > 0);
        }
        if (mrActive()) {
            runMultiRes();
            return;
        }
        finished = true;
        pump();
        if (postRatio() != 1.0) {
            drainEngineToResOut();
            resOut->finish();
        }
    }

    // engine output finalized frames not yet handed to resOut
    long long engineHanded = 0;
    void drainEngineToResOut() {
        while (true) {
            long long fin = finalized();
            if (finished) fin = synthCount * hs + N / 2;
            const int n = static_cast<int>(
                std::min<long long>(fin - engineHanded,
                                    static_cast<long long>(resBuf[0].size())));
            if (n <= 0) return;
            for (int ch = 0; ch < cfg.channels; ++ch)
                wola[ch]->read(engineHanded, n, resPtr[ch]);
            engineHanded += n;
            for (int c = 0; c < cfg.channels; ++c) cPtr[c] = resPtr[c];
            resOut->feed(const_cast<const float* const*>(cPtr.data()), n);
        }
    }

    int availableUser() {
        if (mrActive()) {
            if (!mrDone) return 0;
            return (int)std::min<long long>(
                1 << 30, (long long)mrOut[0].size() - mrReadPos);
        }
        if (postRatio() != 1.0) {
            drainEngineToResOut();
            const int taps = 72;
            return std::max(0, static_cast<int>(
                                   (resOut->pending() - taps) * postRatio()));
        }
        long long f = finalized();
        if (finished) f = synthCount * hs + N / 2;
        return static_cast<int>(std::max(0LL, f - readPos));
    }

    int readUser(float* const* out, int nf) {
        if (mrActive()) {
            if (!mrDone) return 0;
            const long long avail = (long long)mrOut[0].size() - mrReadPos;
            const int n = (int)std::min<long long>(nf, std::max(0LL, avail));
            for (int ch = 0; ch < cfg.channels; ++ch)
                std::memcpy(out[ch], mrOut[ch].data() + mrReadPos,
                            n * sizeof(float));
            mrReadPos += n;
            return n;
        }
        if (postRatio() != 1.0) {
            drainEngineToResOut();
            return resOut->process(out, nf, postRatio());
        }
        const int n = std::min(nf, availableUser());
        for (int ch = 0; ch < cfg.channels; ++ch)
            wola[ch]->read(readPos, n, out[ch]);
        readPos += n;
        return n;
    }

    static float median5(float a, float b, float c, float d, float e) {
        float v[5] = {a, b, c, d, e};
        for (int i = 1; i < 5; ++i) {
            const float key = v[i];
            int j = i - 1;
            while (j >= 0 && v[j] > key) {
                v[j + 1] = v[j];
                --j;
            }
            v[j + 1] = key;
        }
        return v[2];
    }

    // Per-bin noisiness -> phase jitter amount (Damskagg & Valimaki 2017,
    // adapted): tonal bins have a high across-time median relative to the
    // across-frequency median; noisy bins the reverse. Jitter ramps up with
    // the effective stretch (alpha*, includes the pitch factor).
    // Harmonic-product-spectrum F0 estimate on the reference frame, restricted
    // to the speech/singing range. Returns the fundamental bin, or -1 if the
    // frame is not confidently voiced (harmonic-sum peak below a threshold).
    double estimateF0Bin(const AnalysisFrame& R) {
        const int B = N / 2 + 1;
        const double binHz = static_cast<double>(cfg.sampleRate) / N;
        const double f0lo = 70.0, f0hi = 500.0;
        const int lo = std::max(1, static_cast<int>(f0lo / binHz));
        const int hi = std::min(B / 4, static_cast<int>(f0hi / binHz) + 1);
        float maxMag = 0.0f;
        for (int m = 0; m < B; ++m) maxMag = std::max(maxMag, R.mag[m]);
        if (maxMag < 1e-8f) return -1.0;
        double bestScore = 0.0, meanScore = 0.0;
        int bestBin = -1;
        int cnt = 0;
        for (int f = lo; f <= hi; ++f) {
            // sum the first 6 harmonics (product-ish via log-sum for robustness)
            double s = 0.0;
            for (int h = 1; h <= 6; ++h) {
                const int mb = f * h;
                if (mb >= B) break;
                s += R.mag[mb];
            }
            meanScore += s;
            ++cnt;
            if (s > bestScore) { bestScore = s; bestBin = f; }
        }
        meanScore /= std::max(1, cnt);
        // voiced confidence: harmonic sum must dominate the mean substantially
        if (bestBin < 0 || bestScore < 2.2 * meanScore) return -1.0;
        // parabolic refine on the harmonic-sum curve neighbourhood: sub-bin F0
        // localization steadies the fundamental peak p1 in stepVoice (a bin of
        // F0 jitter jumps the whole harmonic stack -> warble).
        auto hsum = [&](int f) {
            double s = 0.0;
            for (int h = 1; h <= 6; ++h) {
                const int mb = f * h;
                if (mb >= B) break;
                s += R.mag[mb];
            }
            return s;
        };
        const double y0 = (bestBin > lo) ? hsum(bestBin - 1) : bestScore;
        const double y1 = bestScore;
        const double y2 = (bestBin < hi) ? hsum(bestBin + 1) : bestScore;
        const double denom = (y0 - 2.0 * y1 + y2);
        double frac = 0.0;
        if (std::abs(denom) > 1e-18) frac = 0.5 * (y0 - y2) / denom;
        frac = std::clamp(frac, -0.5, 0.5);
        return static_cast<double>(bestBin) + frac;
    }

    void pushMagHistory(const AnalysisFrame& R) {
        magHist[histCount % kHist].assign(R.mag.begin(), R.mag.end());
        ++histCount;
    }

    void computeJitter(const AnalysisFrame& R, double effAlpha) {
        const int B = N / 2 + 1;
        const float aTerm =
            0.5f * (std::tanh(4.0f * (static_cast<float>(effAlpha) - 1.5f)) + 1.0f);
        if (histCount < kHist || aTerm < 0.01f) {
            std::fill(jitterAmt.begin(), jitterAmt.end(), 0.0f);
            return;
        }
        const auto& h0 = magHist[(histCount - 1) % kHist];
        const auto& h1 = magHist[(histCount - 2) % kHist];
        const auto& h2 = magHist[(histCount - 3) % kHist];
        const auto& h3 = magHist[(histCount - 4) % kHist];
        const auto& h4 = magHist[(histCount - 5) % kHist];
        // frequency median over ~+-250 Hz (literature: 500 Hz window) —
        // wide enough that even dense harmonic combs stay below 50% fill
        const int half = std::max(8, static_cast<int>(250.0 * N / cfg.sampleRate));
        float maxMag = 0.0f;
        for (int m = 0; m < B; ++m)
            if (R.mag[m] > maxMag) maxMag = R.mag[m];
        const float lvlGate = maxMag * 0.005623f;  // -45 dB: leave the
        // inaudible deterministic floor (tone sidelobes) phase-locked
        std::vector<float> win;
        win.reserve(2 * half + 1);
        for (int m = 0; m < B; ++m) {
            if (R.mag[m] < lvlGate) {
                jitterAmt[m] = 0.0f;
                continue;
            }
            const int lo = std::max(0, m - half);
            const int hi = std::min(B - 1, m + half);
            win.assign(R.mag.begin() + lo, R.mag.begin() + hi + 1);
            const size_t mid = win.size() / 2;
            std::nth_element(win.begin(), win.begin() + mid, win.end());
            const float medF = win[mid];
            const float medT = median5(h0[m], h1[m], h2[m], h3[m], h4[m]);
            const float rn = medF / (medT + 1e-12f);
            // band-pass classification: rn ~ 1 = noise (jitter), rn << 1 =
            // tonal (locked), rn >> 1 = transient (locked, the transient
            // engine owns those bins)
            const float nTerm = 0.25f * (std::tanh(4.0f * (rn - 1.0f)) + 1.0f) *
                                (std::tanh(4.0f * (3.0f - rn)) + 1.0f);
            jitterAmt[m] = nTerm * aTerm;
        }
    }

    void feed(const float* const* in, int nf) {
        if (mrActive()) {
            // buffer the (already pre-resampled) input; processed on finish
            for (int ch = 0; ch < cfg.channels; ++ch)
                mrIn[ch].insert(mrIn[ch].end(), in[ch], in[ch] + nf);
            totalIn += nf;
            return;
        }
        for (int ch = 0; ch < cfg.channels; ++ch) {
            const float* s = in[ch];
            for (int i = 0; i < nf; ++i)
                inRing[ch][static_cast<size_t>((totalIn + i) % inCap)] = s[i];
        }
        totalIn += nf;
        pump();
    }

    // Resample every channel of `in` by `ratio` through `rs` (streaming
    // feed/process/finish), returning the fully drained result.
    std::vector<std::vector<float>> resampleAll(
        SincResampler* rs, const std::vector<std::vector<float>>& in,
        double ratio) {
        std::vector<std::vector<float>> out(cfg.channels);
        const long long n = in.empty() ? 0 : (long long)in[0].size();
        const int CH = 8192;
        std::vector<std::vector<float>> buf(cfg.channels,
                                            std::vector<float>(CH * 2));
        std::vector<const float*> ip(cfg.channels);
        std::vector<float*> op(cfg.channels);
        auto drain = [&]() {
            int made;
            do {
                for (int c = 0; c < cfg.channels; ++c) op[c] = buf[c].data();
                made = rs->process(op.data(), CH * 2, ratio);
                if (made > 0)
                    for (int c = 0; c < cfg.channels; ++c)
                        out[c].insert(out[c].end(), buf[c].begin(),
                                      buf[c].begin() + made);
            } while (made == CH * 2);
        };
        long long fed = 0;
        while (fed < n) {
            const int k = (int)std::min<long long>(CH, n - fed);
            for (int c = 0; c < cfg.channels; ++c) ip[c] = in[c].data() + fed;
            rs->feed(ip.data(), k);
            fed += k;
            drain();
        }
        rs->finish();
        drain();
        return out;
    }

    // Run the whole-signal multi-resolution stretch (Offline opt-in path).
    void runMultiRes() {
        std::vector<MultiResStretch::Scale> scales;
        bool pin = true;
        if (cfg.mode == Config::Mode::Voice) {
            scales = MultiResStretch::voicedScales(alphaUser);
            pin = false;
        } else {
            // content-adaptive: percussion -> single window, tone -> long
            // window, tonal mix -> multi-resolution (detected on channel 0)
            scales = MultiResStretch::autoScales(
                mrIn.empty() ? std::vector<float>() : mrIn[0],
                cfg.sampleRate, &pin, alphaUser);
        }
        MultiResStretch mr(cfg.sampleRate, scales, pin);
        // engine's internal stretch = alphaUser * pitchFactor (the pre/post
        // resamplers convert the pitchFactor part into the actual pitch shift)
        const double engineStretch =
            std::clamp(alphaUser * pitchFactor(), 0.02, 64.0);
        std::vector<std::vector<float>> y = mr.process(mrIn, engineStretch);
        if (postRatio() != 1.0)
            mrOut = resampleAll(resOut.get(), y, postRatio());
        else
            mrOut = std::move(y);
        mrDone = true;
        finished = true;
    }

    void appendZeros(int nf) {
        for (int ch = 0; ch < cfg.channels; ++ch)
            for (int i = 0; i < nf; ++i)
                inRing[ch][static_cast<size_t>((totalIn + i) % inCap)] = 0.0f;
        totalIn += nf;
        padAppended += nf;
    }

    // extract N input samples centered at integer position c (zeros outside)
    void extract(int ch, long long c, float* dst) const {
        const long long start = c - N / 2;
        for (int i = 0; i < N; ++i) {
            const long long p = start + i;
            dst[i] = (p < 0 || p >= totalIn)
                         ? 0.0f
                         : inRing[ch][static_cast<size_t>(p % inCap)];
        }
    }

    // Frame scheduler: activates a frame once its input span is available
    // and runs its stages as their due points are reached. Stage due
    // points are anchored to the frame's own input-availability point and
    // spread over one nominal frame period, so the whole-frame cost never
    // lands in a single feed() call in steady state. Everything here is a
    // pure function of totalIn and the audio content, so scheduling is
    // deterministic and chunk-size independent. Once finished, stages run
    // eagerly and the tail is zero-padded as before.
    void pump() {
        while (true) {
            if (!fActive) {
                const long long cIn = llround(inPos);
                if (cIn + N / 2 > totalIn) {
                    if (!finished) return;
                    // finished: pad zeros so the tail can be rendered
                    if (padAppended > static_cast<long long>((N / 2 + 2 * hs) / std::min(alpha, 1.0)) + N)
                        return;  // tail fully covered
                    appendZeros(hs);
                    continue;
                }
                beginFrame(cIn);
            }
            while (fActive) {
                if (!finished && totalIn < stageDue(fStage)) return;
                runStage();
            }
        }
    }

    int numStages() const { return 3 * cfg.channels + 1; }

    // due point of stage s: fDueBase + (s+1)/S of the spread window
    long long stageDue(int s) const {
        return fDueBase + static_cast<long long>(s + 1) * fSpread / numStages();
    }

    void beginFrame(long long cIn) {
        fActive = true;
        fStage = 0;
        fCIn = cIn;
        fDueBase = cIn + N / 2;
        // spread the stages over one nominal frame period: the last stage
        // is due exactly when the next frame's input span completes (one
        // hop of extra output availability latency, see outputLatency())
        fSpread = std::max<long long>(0, llround(hs / alpha));
        fBestE = -1.0;
        fBest = refCh;
        // per-frame snapshots so all channels see identical parameters
        fDoFormant = formant && pitchSemi != 0.0;
        fP = fDoFormant ? std::pow(2.0, pitchSemi / 12.0) : 1.0;
        // resampled input has detail compressed by p: scale the order
        fEffOrder = static_cast<int>(teOrder * std::max(1.0, 1.0 / fP) + 0.5);
        // Formant-correction strength taper (measured via SQUIM MOS): full
        // correction wins for upshift and mild downshift but over-corrects at
        // deep downshift (-24 st scored best with NO correction). Ramp from
        // full at >= -12 st to zero at -24 st.
        fFmtStrength = 1.0f;
        if (pitchSemi < -12.0)
            fFmtStrength = std::clamp(
                static_cast<float>((pitchSemi + 24.0) / 12.0), 0.0f, 1.0f);
        if (const char* s = std::getenv("PBSHIFT_FMT_STRENGTH"))
            fFmtStrength = static_cast<float>(std::atof(s));
    }

    void runStage() {
        const int C = cfg.channels;
        const int s = fStage++;
        if (s < C) {
            stageAnalyze(s);
        } else if (s < 2 * C) {
            stageFormant(s - C);
        } else if (s == 2 * C) {
            stageDecision();
        } else {
            stageRender(s - 2 * C - 1);
            if (s == 3 * C) {  // frame complete
                ++synthCount;
                inPos += fAdvance;
                fActive = false;
            }
        }
    }

    // Transient detector on consecutive analysis frames of the reference
    // channel: SuperFlux-style max-filtered log flux OR-gated with a
    // broadband rise counter; fills resetMask for rising bins.
    bool detectTransient(const AnalysisFrame& R, long long cIn) {
        const int B = N / 2 + 1;
        std::fill(resetMask.begin(), resetMask.end(), 0);
        if (prevRefMag.empty()) {
            prevRefMag.assign(R.mag.begin(), R.mag.end());
            return false;
        }
        float fluxSum = 0.0f;
        int riseCnt = 0, actCnt = 0;
        float maxMag = 0.0f;
        for (int m = 0; m < B; ++m)
            if (R.mag[m] > maxMag) maxMag = R.mag[m];
        const float floor_ = std::max(1e-7f, 1e-4f * maxMag);
        for (int m = 0; m < B; ++m) {
            // 3-bin max filter on previous frame (vibrato tolerance)
            float pm = prevRefMag[m];
            if (m > 0 && prevRefMag[m - 1] > pm) pm = prevRefMag[m - 1];
            if (m + 1 < B && prevRefMag[m + 1] > pm) pm = prevRefMag[m + 1];
            const float cm = R.mag[m];
            if (cm > floor_ || pm > floor_) {
                ++actCnt;
                const float d = std::log(cm + floor_) - std::log(pm + floor_);
                if (d > 0.0f) fluxSum += d;
                if (cm > 1.41f * pm) {  // +3 dB rise
                    ++riseCnt;
                    resetMask[m] = 1;
                }
            }
        }
        lastFlux = 0.0f;
        lastRiseFrac = 0.0f;
        // mag^2-weighted group delay of the rising bins = where the incoming
        // event sits relative to the window center (+ = still ahead)
        double tw = 0.0, twSum = 0.0;
        for (int m = 0; m < B; ++m) {
            if (!resetMask[m]) continue;
            const double w2 = static_cast<double>(R.mag[m]) * R.mag[m];
            tw += w2 * R.tau[m];
            twSum += w2;
        }
        tauMean = twSum > 0.0 ? static_cast<float>(tw / twSum) : 0.0f;

        const float flux = actCnt ? fluxSum / actCnt : 0.0f;
        const float riseFrac = actCnt ? static_cast<float>(riseCnt) / actCnt : 0.0f;
        const bool refractory =
            (cIn - lastOnsetIn) < 0.05 * cfg.sampleRate;  // 50 ms
        const bool onset = !refractory &&
                           ((flux > 0.22f && flux > 1.1f * prevFlux) ||
                            riseFrac > 0.35f) &&
                           riseFrac > 0.12f;
        lastFlux = flux;
        lastRiseFrac = riseFrac;
        prevFlux = flux;
        prevRefMag.assign(R.mag.begin(), R.mag.end());
        if (!onset && pendCount == 0)
            std::fill(resetMask.begin(), resetMask.end(), 0);
        if (onset && debugOnsets)
            std::fprintf(stderr, "onset@in=%.3fs flux=%.3f rise=%.3f tau=%.0f\n",
                         cIn / (double)cfg.sampleRate, flux, riseFrac, tauMean);
        return onset;
    }
    bool debugOnsets = std::getenv("PBSHIFT_DEBUG_ONSETS") != nullptr;
    bool noTransient = std::getenv("PBSHIFT_NO_TRANSIENT") != nullptr;
    bool noPin = std::getenv("PBSHIFT_NO_PIN") != nullptr;
    bool forceVoice = std::getenv("PBSHIFT_VOICE") != nullptr;
    // Coherence-locked identity kernel (rebuild each peak region's phase from
    // the reassignment group delay tau, scaled by alpha, instead of copying the
    // analysis phase verbatim). Cures the hop-synchronous comb/"fine DelayEcho"
    // on voice. Opt-in; also used as the non-voiced fallback inside Voice mode.
    bool forceCoherent = std::getenv("PBSHIFT_COHERENT") != nullptr;
    // measured on the current metric set: phase jitter hurt attack/LTAS and
    // bought nothing on warble — opt-in only until real Noise Morphing lands
    bool noJitter = std::getenv("PBSHIFT_JITTER") == nullptr;
    bool usePghi = std::getenv("PBSHIFT_PHASE") &&
                   !std::strcmp(std::getenv("PBSHIFT_PHASE"), "pghi");

    // ---- frame stages (formerly one monolithic synthesizeFrame) ------

    // analyze one channel; accumulate reference election (sticky energy
    // argmax); the election closes on the last channel
    void stageAnalyze(int ch) {
        extract(ch, fCIn, frameBuf.data());
        fe[ch]->analyze(frameBuf.data(), frames[ch]);
        double e = 0.0;
        for (float m : frames[ch].mag) e += static_cast<double>(m) * m;
        if (e > fBestE) {
            fBestE = e;
            fBest = ch;
        }
        if (ch == refCh) fBestE *= 1.5;  // hysteresis: favour current ref
        if (ch == cfg.channels - 1) refCh = fBest;
    }

    // ---- formant preservation (M6), one channel per stage ------------
    // The front resampler moved formants by p; restore the original
    // envelope: desired(f) = env_in(f*p).
    void stageFormant(int ch) {
        if (!fDoFormant) return;
        AnalysisFrame& F = frames[ch];
        tenv[ch]->compute(F.mag.data(), envLog, fEffOrder);
        // temporal smoothing kills frame-to-frame gain flutter
        if (envLambda > 0.0f) {
            auto& s = envSmooth[ch];
            if (s.size() != envLog.size()) {
                s = envLog;
            } else {
                for (size_t m = 0; m < s.size(); ++m)
                    s[m] = envLambda * s[m] + (1.0f - envLambda) * envLog[m];
                envLog = s;
            }
        }
        formantGains(envLog, fP, fmtGain, fFmtStrength, tenv[ch]->lastF0Bin());
        const int B = N / 2 + 1;
        for (int m = 0; m < B; ++m) F.mag[m] *= fmtGain[m];
    }

    // ---- atomic decision stage ---------------------------------------
    // Transient detect + hold/fire/pin + advance/debt decision + phase
    // engine, in the exact per-frame sequential order the state machines
    // (pendCount/pinRemain/debt/lastOnsetIn/prevFlux, magHist, envSmooth,
    // pghi) were designed around. Decides fAdvance and fCOut.
    void stageDecision() {
        const long long cIn = fCIn;
        // ---- transient handling (Roebel hold -> fire -> pin) ---------
        // hold: rising bins of a coming attack keep previous magnitudes so
        //       pre-fire frames don't pre-echo the attack
        // fire: when the attack reaches the window center, synchronized
        //       phase reset places it once
        // pin:  local ratio 1 across all frames overlapping the attack so
        //       every frame agrees on its output position (no doubling)
        const bool rise = noTransient ? false : detectTransient(frames[refCh], cIn);
        const float Ce = 0.044f * N;  // Roebel center-of-gravity threshold
        bool fire = false;
        if (pendCount > 0) {
            // already holding: fire when event reaches center or timeout
            if (tauMean <= Ce || pendCount >= 3) {
                fire = true;
                pendCount = 0;
            } else {
                ++pendCount;
            }
        } else if (rise) {
            if (tauMean > Ce && std::abs(debt) < 1.5 * hs) {
                pendCount = 1;  // event still ahead: hold this frame
            } else {
                fire = true;    // event at/past center: place it now
            }
        }
        if (pendCount > 0) {
            // hold rising bins at their pre-event magnitudes (per channel)
            const int B = N / 2 + 1;
            for (int ch = 0; ch < cfg.channels; ++ch) {
                auto& pm = prevMagCh[ch];
                if (pm.size() == static_cast<size_t>(B))
                    for (int m = 0; m < B; ++m)
                        if (resetMask[m]) frames[ch].mag[m] = pm[m];
            }
        }
        const double nominal = hs / alpha;
        double advance = nominal;
        float localAlpha = static_cast<float>(alpha);
        const bool onset = fire;
        if (fire) {
            lastOnsetIn = static_cast<double>(cIn);
            // Ratio pinning is for percussive material: on voice it warps
            // local timing around every consonant (measured +2..4 dB
            // envelope LSD at stretch<1). Pin only when (a) the material
            // context allows it and (b) the event itself is a strong
            // broadband hit (speech consonants measure flux <~2, drum and
            // click hits ~5-7).
            const bool voiceish =
                cfg.mode == Config::Mode::Voice ||
                (cfg.mode == Config::Mode::Auto && formant) || noPin;
            const bool strongHit = lastFlux > 2.5f && lastRiseFrac > 0.5f;
            // NOTE: gating the pin to expansion-only was measured net-negative
            // (clicks 0.5x LTAS 0.97 -> 1.71: the phase-reset-only fallback
            // leaks MORE than the pinned path). The pin helps at every ratio
            // it fires; the residual LTAS cost at large expansion is the
            // inherent price of attack-sharpness preservation.
            if (!voiceish && strongHit && std::abs(debt) < 4.0 * hs)
                pinRemain = N / hs;  // this frame + the overlapping ones
        }
        if (pinRemain > 0) {
            advance = hs;
            debt += advance - nominal;
            localAlpha = 1.0f;
            --pinRemain;
        } else if (std::abs(debt) > 1e-6 && pendCount == 0) {
            // repay timing debt over ~100 ms of non-transient audio
            const double framesPerRepay =
                std::max(1.0, 0.1 * cfg.sampleRate / hs);
            double repay = debt / framesPerRepay;
            const double cap = 0.5 * nominal;  // keep ratio swing bounded
            repay = std::clamp(repay, -cap, cap);
            advance = nominal - repay;
            debt -= repay;
        }
        // store post-hold magnitudes for the next frame's hold reference;
        // while holding, the detector's reference must also stay pre-event
        // so the fire frame still sees the full rising set
        for (int ch = 0; ch < cfg.channels; ++ch)
            prevMagCh[ch].assign(frames[ch].mag.begin(), frames[ch].mag.end());
        if (pendCount > 0)
            prevRefMag.assign(frames[refCh].mag.begin(),
                              frames[refCh].mag.end());

        fCOut = synthCount * hs;
        const bool transientActive = onset || pendCount > 0 || pinRemain > 0;
        pushMagHistory(frames[refCh]);
        if (!noJitter && !transientActive)
            computeJitter(frames[refCh], alpha);
        else
            std::fill(jitterAmt.begin(), jitterAmt.end(), 0.0f);
        // Voice mode: shape-invariant harmonic-locked phase on voiced frames
        // (transient frames still take the synchronized reset via stepLocked).
        // Widened to alphaUser in [0.9, 2.5] to cover 2x time-stretch: the hop-
        // synchronous comb ("fine DelayEcho") that identity-lock leaves on
        // stretched voice is exactly what harmonic locking removes. Non-voiced
        // frames within Voice mode take the coherence-locked kernel (not plain
        // stepLocked) so the whole utterance stays comb-free. Only fires in
        // explicit Voice mode / PBSHIFT_VOICE, so the Auto default is unaffected.
        const bool voiceMode =
            (cfg.mode == Config::Mode::Voice || forceVoice) && !onset &&
            alphaUser >= 0.9 && alphaUser <= 2.5;
        double f0Bin = -1.0;
        if (voiceMode) f0Bin = estimateF0Bin(frames[refCh]);
        if (voiceMode && f0Bin > 0.0) {
            pghi->stepVoice(frames[refCh], hs, f0Bin, synthPhase);
        } else if (usePghi) {
            pghi->step(frames[refCh], hs, localAlpha,
                       static_cast<uint64_t>(synthCount),
                       onset ? resetMask.data() : nullptr, synthPhase);
        } else if (forceCoherent || voiceMode) {
            // coherence-locked identity kernel: explicit opt-in (PBSHIFT_COHERENT
            // on any mode), or the non-voiced fallback inside Voice mode.
            pghi->stepLockedCoherent(frames[refCh], hs, localAlpha,
                                     onset ? resetMask.data() : nullptr,
                                     synthPhase);
        } else {
            pghi->stepLocked(frames[refCh], hs,
                             onset ? resetMask.data() : nullptr, synthPhase,
                             noJitter ? nullptr : jitterAmt.data(),
                             static_cast<uint64_t>(synthCount) | (1ull << 40));
        }
        fAdvance = advance;
    }

    // render one channel: spectrum assembly + iFFT + WOLA accumulate
    void stageRender(int ch) {
        const int B = N / 2 + 1;
        const AnalysisFrame& R = frames[refCh];
        const AnalysisFrame& F = frames[ch];
        if (ch == refCh) {
            for (int m = 0; m < B; ++m)
                Y[m] = std::polar(F.mag[m], synthPhase[m]);
        } else {
            // copy input inter-channel phase delta verbatim
            for (int m = 0; m < B; ++m) {
                std::complex<float> rel = F.X[m] * std::conj(R.X[m]);
                const float rm = std::abs(rel);
                if (rm > 1e-20f)
                    rel /= rm;
                else
                    rel = {1.0f, 0.0f};
                Y[m] = F.mag[m] * std::polar(1.0f, synthPhase[m]) * rel;
            }
        }
        // real-FFT edge bins must stay real
        Y[0] = {std::abs(Y[0]) * (std::cos(std::arg(Y[0])) < 0 ? -1.0f : 1.0f), 0.0f};
        Y[B - 1] = {std::abs(Y[B - 1]) * (std::cos(std::arg(Y[B - 1])) < 0 ? -1.0f : 1.0f), 0.0f};
        wola[ch]->addFrame(Y.data(), fCOut);
    }

    long long finalized() const {
        // a sample is final when no future frame can touch it
        const long long nextCenter = synthCount * hs;
        return std::max(0LL, nextCenter - N / 2);
    }
};

Stretcher::Stretcher() : impl_(new Impl) {}
Stretcher::~Stretcher() = default;

void Stretcher::configure(const Config& cfg) { impl_->configure(cfg); }
void Stretcher::reset() { impl_->resetState(); }

void Stretcher::setTimeStretch(double ratio) {
    impl_->alphaUser = std::clamp(ratio, 0.05, 20.0);
    impl_->updateAlpha();
}
void Stretcher::setPitchSemitones(double s) {
    impl_->pitchSemi = std::clamp(s, -24.0, 24.0);
    impl_->updateAlpha();
}
void Stretcher::setFormantPreserve(bool e) { impl_->formant = e; }

void Stretcher::feed(const float* const* in, int frames) {
    impl_->feedUser(in, frames);
}

void Stretcher::finish() { impl_->finishUser(); }

int Stretcher::available() const {
    return const_cast<Impl*>(impl_.get())->availableUser();
}

int Stretcher::read(float* const* out, int frames) {
    return impl_->readUser(out, frames);
}

int Stretcher::inputLatency() const { return impl_->N / 2; }
// + one extra hop: the staged frame pipeline finishes a frame up to one
// nominal frame period after its input span completes (output content is
// unchanged; only availability shifts by at most hs output samples).
int Stretcher::outputLatency() const { return impl_->N / 2 + 2 * impl_->hs; }

}  // namespace pbshift
