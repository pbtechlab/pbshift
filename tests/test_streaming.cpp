// Streaming determinism gates:
//  1. chunk-size independence: feeding 64 vs 8192 samples per call must
//     produce bit-identical output (host-buffer independence)
//  2. re-render determinism: two runs, identical bits
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "pbshift/pbshift.h"

using pbshift::Config;
using pbshift::Stretcher;
static constexpr double kPi = 3.1415926535897932384626433832795;

static std::vector<std::vector<float>> render(
    const std::vector<std::vector<float>>& x, int chunk,
    double stretch, double pitch, int sr,
    Config::Mode mode = Config::Mode::Auto,
    Config::Tier tier = Config::Tier::StudioRT) {
    const int ch = static_cast<int>(x.size());
    Config cfg;
    cfg.sampleRate = sr;
    cfg.channels = ch;
    cfg.mode = mode;
    cfg.tier = tier;
    Stretcher st;
    st.configure(cfg);
    st.setTimeStretch(stretch);
    st.setPitchSemitones(pitch);
    const long long n = static_cast<long long>(x[0].size());
    const long long target = llround(n * stretch);
    std::vector<std::vector<float>> y(ch, std::vector<float>(target, 0.0f));
    std::vector<const float*> ip(ch);
    std::vector<float*> op(ch);
    long long fed = 0, got = 0;
    while (fed < n) {
        const int k = static_cast<int>(std::min<long long>(chunk, n - fed));
        for (int c = 0; c < ch; ++c) ip[c] = x[c].data() + fed;
        st.feed(ip.data(), k);
        fed += k;
        int avail = st.available();
        while (avail > 0 && got < target) {
            const int want = static_cast<int>(
                std::min<long long>(std::min(avail, chunk), target - got));
            for (int c = 0; c < ch; ++c) op[c] = y[c].data() + got;
            const int r = st.read(op.data(), want);
            if (r <= 0) break;
            got += r;
            avail = st.available();
        }
    }
    st.finish();
    while (got < target) {
        const int avail = st.available();
        if (avail <= 0) break;
        const int want =
            static_cast<int>(std::min<long long>(avail, target - got));
        for (int c = 0; c < ch; ++c) op[c] = y[c].data() + got;
        const int r = st.read(op.data(), want);
        if (r <= 0) break;
        got += r;
    }
    if (got != target) {
        std::fprintf(stderr, "incomplete render: got=%lld target=%lld\n",
                     got, target);
        std::exit(2);
    }
    return y;
}

// Drain the public API completely rather than clipping to a caller-side
// target.  The near-unity offline renderer owns exact length, so this helper
// catches both short output and an accidental extra tail.
static std::vector<std::vector<float>> renderAll(
    const std::vector<std::vector<float>>& x, int chunk,
    double stretch, int sr, Config::Tier tier,
    Config::Mode mode = Config::Mode::Auto) {
    const int ch = static_cast<int>(x.size());
    Config cfg;
    cfg.sampleRate = sr;
    cfg.channels = ch;
    cfg.tier = tier;
    cfg.mode = mode;
    Stretcher st;
    st.configure(cfg);
    st.setTimeStretch(stretch);
    st.setPitchSemitones(0.0);

    std::vector<std::vector<float>> y(ch);
    std::vector<const float*> ip(ch);
    std::vector<std::vector<float>> scratch(ch, std::vector<float>(8192));
    std::vector<float*> op(ch);
    for (int c = 0; c < ch; ++c) op[c] = scratch[c].data();
    auto drain = [&]() {
        while (st.available() > 0) {
            const int want = std::min(8192, st.available());
            const int got = st.read(op.data(), want);
            if (got <= 0) break;
            for (int c = 0; c < ch; ++c)
                y[c].insert(y[c].end(), scratch[c].begin(),
                            scratch[c].begin() + got);
        }
    };

    const long long n = static_cast<long long>(x[0].size());
    long long fed = 0;
    while (fed < n) {
        const int k = static_cast<int>(std::min<long long>(chunk, n - fed));
        for (int c = 0; c < ch; ++c) ip[c] = x[c].data() + fed;
        st.feed(ip.data(), k);
        fed += k;
        drain();
    }
    st.finish();
    drain();
    return y;
}

static double risingCrossingHz(const std::vector<float>& x, int sr,
                               double beginFrac = 0.15,
                               double endFrac = 0.85) {
    const size_t begin = static_cast<size_t>(x.size() * beginFrac);
    const size_t end = static_cast<size_t>(x.size() * endFrac);
    long long count = 0;
    for (size_t i = std::max<size_t>(1, begin); i < end; ++i)
        if (x[i - 1] <= 0.0f && x[i] > 0.0f) ++count;
    return count * (double)sr / std::max<size_t>(1, end - begin);
}

static double rmsEnvelopeRangeDb(const std::vector<float>& x, int sr) {
    const int window = std::max(16, sr / 10);  // 100 ms
    const int hop = std::max(8, sr / 50);      // 20 ms
    const int begin = static_cast<int>(x.size() / 5);
    const int end = static_cast<int>(x.size() * 4 / 5);
    double lo = 1e30, hi = 0.0;
    for (int p = begin; p + window <= end; p += hop) {
        double e = 0.0;
        for (int i = 0; i < window; ++i)
            e += static_cast<double>(x[p + i]) * x[p + i];
        const double rms = std::sqrt(e / window + 1e-30);
        lo = std::min(lo, rms);
        hi = std::max(hi, rms);
    }
    return hi > 0.0 && lo < 1e29 ? 20.0 * std::log10(hi / lo) : 0.0;
}

static std::vector<size_t> eventClusters(const std::vector<float>& x,
                                         float threshold, int mergeGap) {
    std::vector<size_t> peaks;
    size_t i = 0;
    while (i < x.size()) {
        if (std::abs(x[i]) < threshold) {
            ++i;
            continue;
        }
        size_t best = i;
        float level = std::abs(x[i]);
        size_t last = i;
        ++i;
        while (i < x.size() && i <= last + static_cast<size_t>(mergeGap)) {
            if (std::abs(x[i]) >= threshold) {
                last = i;
                if (std::abs(x[i]) > level) {
                    level = std::abs(x[i]);
                    best = i;
                }
            }
            ++i;
        }
        peaks.push_back(best);
    }
    return peaks;
}

int main() {
    const int SR = 48000, ch = 2;
    const int n = SR * 2;
    std::vector<std::vector<float>> x(ch, std::vector<float>(n));
    for (int i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / SR;
        x[0][i] = 0.5f * std::sin(2 * 3.14159265f * 220 * t) +
                  0.2f * std::sin(2 * 3.14159265f * 3001 * t);
        x[1][i] = 0.4f * std::sin(2 * 3.14159265f * 330 * t + 0.5f);
        if (i % 24000 == 0) x[0][i] += 0.8f;  // clicks
    }

    struct Condition {
        double stretch;
        double pitch;
        Config::Mode mode;
        const char* name;
    };
    const Condition conditions[] = {
        {1.7, 0.0, Config::Mode::Auto, "auto"},
        {0.6, 0.0, Config::Mode::Auto, "auto"},
        {1.0, 7.0, Config::Mode::Auto, "auto"},
        {1.0, -7.0, Config::Mode::Auto, "auto"},
        // Music mode has a separate fine-hop/short-window path. Keep it under
        // the same determinism gates so mode-specific fixes cannot weaken the
        // host-buffer-independence contract.
        {2.0, 0.0, Config::Mode::Music, "music"},
        {1.0, 12.0, Config::Mode::Music, "music"},
        {2.0, 0.0, Config::Mode::Rhythm, "rhythm"},
        {2.0, 0.0, Config::Mode::Voice, "voice"},
    };

    int fails = 0;
    for (const auto& cond : conditions) {
        auto a = render(x, 64, cond.stretch, cond.pitch, SR, cond.mode);
        auto b = render(x, 8192, cond.stretch, cond.pitch, SR, cond.mode);
        size_t diff = 0;
        for (int c = 0; c < ch; ++c)
            for (size_t i = 0; i < a[c].size(); ++i)
                if (a[c][i] != b[c][i]) ++diff;
        auto a2 = render(x, 64, cond.stretch, cond.pitch, SR, cond.mode);
        size_t diff2 = 0;
        for (int c = 0; c < ch; ++c)
            for (size_t i = 0; i < a[c].size(); ++i)
                if (a[c][i] != a2[c][i]) ++diff2;
        std::printf("%s mode=%s stretch=%.2f pitch=%+.0f "
                    "chunk-independence diffs=%zu "
                    "rerender diffs=%zu\n",
                    (diff == 0 && diff2 == 0) ? "PASS" : "FAIL",
                    cond.name, cond.stretch, cond.pitch, diff, diff2);
        if (diff || diff2) ++fails;
    }

    // The Music/Rhythm path is designed around a hop half the Auto-mode hop.
    // outputLatency() exposes that scheduler choice through the public API and
    // catches accidental reset to N/4 after the normal parameter setters.
    Config autoCfg;
    autoCfg.sampleRate = SR;
    autoCfg.channels = ch;
    Config musicCfg = autoCfg;
    musicCfg.mode = Config::Mode::Music;
    Config rhythmCfg = autoCfg;
    rhythmCfg.mode = Config::Mode::Rhythm;
    Stretcher autoSt, musicSt, rhythmSt;
    autoSt.configure(autoCfg);
    musicSt.configure(musicCfg);
    rhythmSt.configure(rhythmCfg);
    const int musicConfiguredLatency = musicSt.outputLatency();
    autoSt.setTimeStretch(2.0);
    musicSt.setTimeStretch(2.0);
    rhythmSt.setTimeStretch(2.0);
    const int musicStretchLatency = musicSt.outputLatency();
    autoSt.setPitchSemitones(0.0);
    musicSt.setPitchSemitones(0.0);
    rhythmSt.setPitchSemitones(0.0);
    const bool fineHop =
        autoSt.outputLatency() == 2 * autoSt.inputLatency() &&
        musicSt.outputLatency() ==
            musicSt.inputLatency() + musicSt.inputLatency() / 2 &&
        musicConfiguredLatency == musicStretchLatency &&
        musicStretchLatency == musicSt.outputLatency() &&
        rhythmSt.outputLatency() == musicSt.outputLatency();
    std::printf("%s fine-hop latency auto=%d music=%d rhythm=%d\n",
                fineHop ? "PASS" : "FAIL", autoSt.outputLatency(),
                musicSt.outputLatency(), rhythmSt.outputLatency());
    if (!fineHop) ++fails;

    // Ratios changed during a stream are intentionally not allowed to retime
    // the in-flight grid. reset()/reconfigure() must, however, derive the next
    // stream's compression-safe hop from the retained parameter values.
    Stretcher reused;
    reused.configure(autoCfg);
    std::vector<const float*> reuseIn(ch);
    for (int c = 0; c < ch; ++c) reuseIn[c] = x[c].data();
    reused.feed(reuseIn.data(), 8192);
    reused.setTimeStretch(0.25);
    const int inFlightLatency = reused.outputLatency();
    reused.reset();
    const int resetLatency = reused.outputLatency();
    reused.configure(autoCfg);
    const int reconfiguredLatency = reused.outputLatency();
    const bool resetHop = resetLatency < inFlightLatency &&
                          reconfiguredLatency == resetLatency;
    std::printf("%s reset/reconfigure hop in-flight=%d reset=%d reconfig=%d\n",
                resetHop ? "PASS" : "FAIL", inFlightLatency, resetLatency,
                reconfiguredLatency);
    if (!resetHop) ++fails;

    // Offline near-unity stretch is content-adaptive WSOLA/MultiRes: exact
    // target lengths, host-buffer independence (including a one-shot feed),
    // finite/bounded output, and one shared trajectory for all channels.
    const int nearN = SR * 3;
    std::vector<std::vector<float>> nearX(ch, std::vector<float>(nearN));
    for (int i = 0; i < nearN; ++i) {
        const double t = static_cast<double>(i) / SR;
        const float l = static_cast<float>(
            0.43 * std::sin(2.0 * kPi * 440.0 * t) +
            0.17 * std::sin(2.0 * kPi * 1321.0 * t + 0.2) +
            0.05 * std::sin(2.0 * kPi * 73.0 * t));
        nearX[0][i] = l;
        nearX[1][i] = -0.375f * l;
        if (i % 16000 == 0) {
            nearX[0][i] += 0.25f;
            nearX[1][i] = -0.375f * nearX[0][i];
        }
    }

    const double nearRatios[] = {0.98, 1.01, 1.02, 1.05};
    for (double ratio : nearRatios) {
        auto small = renderAll(nearX, 37, ratio, SR, Config::Tier::Offline);
        auto block = renderAll(nearX, 8192, ratio, SR, Config::Tier::Offline);
        auto one = renderAll(nearX, nearN, ratio, SR, Config::Tier::Offline);
        const size_t target = static_cast<size_t>(llround(nearN * ratio));
        size_t diffs = 0;
        bool finite = true;
        double peak = 0.0;
        double stereoError = 0.0;
        const bool lengths = small[0].size() == target &&
                             block[0].size() == target &&
                             one[0].size() == target;
        if (lengths) {
            for (int c = 0; c < ch; ++c)
                for (size_t i = 0; i < target; ++i) {
                    if (small[c][i] != block[c][i] ||
                        small[c][i] != one[c][i])
                        ++diffs;
                    finite = finite && std::isfinite(small[c][i]);
                    peak = std::max(peak, std::abs((double)small[c][i]));
                }
            for (size_t i = 0; i < target; ++i)
                stereoError = std::max(
                    stereoError,
                    std::abs((double)small[1][i] + 0.375 * small[0][i]));
        }
        const bool pass = lengths && diffs == 0 && finite && peak <= 0.86 &&
                          stereoError < 1e-3;
        std::printf("%s near-unity %.2fx length=%zu diffs=%zu "
                    "peak=%.6f stereo-error=%.3g\n",
                    pass ? "PASS" : "FAIL", ratio,
                    small[0].size(), diffs, peak, stereoError);
        if (!pass) ++fails;
    }

    // Unity in Offline tier must be a literal identity, not merely a small
    // numerical null through a reconstruction path.
    auto identity = renderAll(nearX, 777, 1.0, SR, Config::Tier::Offline);
    size_t identityDiffs = 0;
    for (int c = 0; c < ch; ++c)
        for (size_t i = 0; i < nearX[c].size(); ++i)
            if (identity[c][i] != nearX[c][i]) ++identityDiffs;
    const bool identityPass = identity[0].size() == nearX[0].size() &&
                              identityDiffs == 0;
    std::printf("%s offline unity bit-identity diffs=%zu\n",
                identityPass ? "PASS" : "FAIL", identityDiffs);
    if (!identityPass) ++fails;

    // Pure time stretch must not turn into resampling.  A 440 Hz tone remains
    // 440 Hz at the largest near-unity ratio (zero-crossing resolution here is
    // substantially tighter than one musical cent).
    std::vector<std::vector<float>> tone(ch,
                                         std::vector<float>(SR * 2, 0.0f));
    for (int i = 0; i < SR * 2; ++i) {
        tone[0][i] = static_cast<float>(
            0.5 * std::sin(2.0 * kPi * 440.0 * i / SR));
        tone[1][i] = tone[0][i];
    }
    auto stretchedTone =
        renderAll(tone, 4093, 1.05, SR, Config::Tier::Offline);
    long long rising = 0;
    for (size_t i = 1; i < stretchedTone[0].size(); ++i)
        if (stretchedTone[0][i - 1] <= 0.0f &&
            stretchedTone[0][i] > 0.0f)
            ++rising;
    const double measuredHz =
        rising * (double)SR / stretchedTone[0].size();
    const bool pitchPass = std::abs(measuredHz - 440.0) < 0.6;
    std::printf("%s near-unity pitch preservation f0=%.6f Hz\n",
                pitchPass ? "PASS" : "FAIL", measuredHz);
    if (!pitchPass) ++fails;

    std::vector<std::vector<float>> empty(ch);
    auto emptyOut =
        renderAll(empty, 128, 1.02, SR, Config::Tier::Offline);
    std::vector<std::vector<float>> tiny(ch, std::vector<float>(37));
    for (int i = 0; i < 37; ++i) {
        tiny[0][i] = static_cast<float>((i - 18) / 37.0);
        tiny[1][i] = -tiny[0][i];
    }
    auto tinyOut =
        renderAll(tiny, 37, 1.05, SR, Config::Tier::Offline);
    const int shortToneN = static_cast<int>(SR * 0.045);
    std::vector<std::vector<float>> shortTone(
        ch, std::vector<float>(shortToneN, 0.0f));
    for (int i = 0; i < shortToneN; ++i)
        shortTone[0][i] = shortTone[1][i] = static_cast<float>(
            0.4 * std::sin(2.0 * kPi * 1000.0 * i / SR));
    auto shortToneOut = renderAll(shortTone, 251, 1.05, SR,
                                  Config::Tier::Offline,
                                  Config::Mode::Rhythm);
    // Ignore the spectral renderer's unavoidable boundary transient; the
    // sustained centre must retain pitch rather than behave like resampling.
    const double shortToneHz = risingCrossingHz(shortToneOut[0], SR);
    bool tinyFinite = true;
    for (const auto& channel : tinyOut)
        for (float s : channel) tinyFinite = tinyFinite && std::isfinite(s);
    for (const auto& channel : shortToneOut)
        for (float s : channel) tinyFinite = tinyFinite && std::isfinite(s);
    const bool edgePass = emptyOut[0].empty() && tinyFinite &&
                          tinyOut[0].size() ==
                              static_cast<size_t>(llround(37 * 1.05)) &&
                          shortToneOut[0].size() == static_cast<size_t>(
                              llround(shortToneN * 1.05)) &&
                          std::abs(shortToneHz - 1000.0) < 30.0;
    std::printf("%s near-unity empty/short edge lengths=%zu/%zu "
                "short-f0=%.2fHz\n",
                edgePass ? "PASS" : "FAIL", emptyOut[0].size(),
                tinyOut[0].size(), shortToneHz);
    if (!edgePass) ++fails;

    // Endpoint-coded short/normal clips and adversarial impulses.  Every event
    // must survive once, in order, near its uniformly scaled time; protected
    // head/tail regions remain literal source samples.
    bool markerPass = true;
    double markerWorstMs = 0.0;
    for (double seconds : {0.25, 1.0}) {
        const int markerN = static_cast<int>(SR * seconds);
        std::vector<std::vector<float>> marker(ch,
                                               std::vector<float>(markerN, 0.0f));
        const int markerPos[] = {markerN / 4, markerN / 2, 3 * markerN / 4};
        for (int p : markerPos) marker[0][p] = marker[1][p] = 1.0f;
        for (double ratio : {0.98, 1.05}) {
            auto y = renderAll(marker, markerN, ratio, SR,
                               Config::Tier::Offline,
                               Config::Mode::Rhythm);
            const auto events = eventClusters(y[0], 0.20f, SR / 200);
            markerPass = markerPass && events.size() == 3;
            if (events.size() == 3) {
                for (int i = 0; i < 3; ++i) {
                    const double err = std::abs(
                        (double)events[static_cast<size_t>(i)] -
                        markerPos[i] * ratio);
                    markerWorstMs = std::max(markerWorstMs, 1000.0 * err / SR);
                    markerPass = markerPass && err <= SR * 0.012;
                }
            }
        }

        std::vector<std::vector<float>> coded(ch,
                                              std::vector<float>(markerN));
        uint32_t state = 0x13579bdu;
        for (int i = 0; i < markerN; ++i) {
            state = state * 1664525u + 1013904223u;
            const float v = (static_cast<int>((state >> 8) & 0xffff) -
                             32768) * (0.2f / 32768.0f);
            coded[0][i] = v;
            coded[1][i] = -0.5f * v;
        }
        auto codedY = renderAll(coded, 333, 1.02, SR,
                                Config::Tier::Offline,
                                Config::Mode::Rhythm);
        for (int c = 0; c < ch; ++c)
            for (int i = 0; i < 128; ++i) {
                markerPass = markerPass && codedY[c][i] == coded[c][i];
                markerPass = markerPass &&
                    codedY[c][codedY[c].size() - 128 + i] ==
                        coded[c][coded[c].size() - 128 + i];
            }
    }
    std::printf("%s endpoint/impulse markers worst=%.3f ms\n",
                markerPass ? "PASS" : "FAIL", markerWorstMs);
    if (!markerPass) ++fails;

    // A loud decorrelated broadband channel must not hide a -40 dB low-bass
    // channel from the tonal MultiRes safeguard. The bass keeps its
    // pitch/envelope while the loud reference channel retains sane RMS.
    bool bassPass = true;
    double worstBassHz = 0.0, worstBassEnv = 0.0, worstLoudDb = 0.0;
    for (double f0 : {50.0, 73.0, 80.0}) {
        const int bassN = SR * 3;
        std::vector<std::vector<float>> bass(ch,
                                             std::vector<float>(bassN, 0.0f));
        uint32_t state = 0x2468aceu;
        for (int i = 0; i < bassN; ++i) {
            state = state * 1103515245u + 12345u;
            bass[0][i] = (static_cast<int>((state >> 9) & 0x7fff) -
                           16384) * (0.35f / 16384.0f);
            bass[1][i] = static_cast<float>(
                0.0035 * std::sin(2.0 * kPi * f0 * i / SR));
        }
        for (double ratio : {0.97, 0.98, 1.02, 1.05}) {
            auto y = renderAll(bass, 4096, ratio, SR,
                               Config::Tier::Offline,
                               Config::Mode::Rhythm);
            const double hz = risingCrossingHz(y[1], SR);
            const double env = rmsEnvelopeRangeDb(y[1], SR);
            double inEnergy = 0.0, outEnergy = 0.0;
            for (float s : bass[0]) inEnergy += static_cast<double>(s) * s;
            for (float s : y[0]) outEnergy += static_cast<double>(s) * s;
            const double loudDb = 10.0 * std::log10(
                (outEnergy / y[0].size() + 1e-30) /
                (inEnergy / bass[0].size() + 1e-30));
            std::printf("  bass %.0fHz %.2fx -> %.4fHz env=%.3fdB\n",
                        f0, ratio, hz, env);
            worstBassHz = std::max(worstBassHz, std::abs(hz - f0));
            worstBassEnv = std::max(worstBassEnv, env);
            worstLoudDb = std::max(worstLoudDb, std::abs(loudDb));
            bassPass = bassPass && std::abs(hz - f0) < 0.8 && env < 2.0 &&
                       std::abs(loudDb) < 1.0;

            // pbCosmo stores its Multi selection as Offline Music. A
            // representative case must traverse that public mode too.
            if (f0 == 50.0 && ratio == 1.02) {
                auto musicY = renderAll(bass, 4096, ratio, SR,
                                        Config::Tier::Offline,
                                        Config::Mode::Music);
                const double musicHz = risingCrossingHz(musicY[1], SR);
                const double musicEnv = rmsEnvelopeRangeDb(musicY[1], SR);
                bassPass = bassPass && std::abs(musicHz - f0) < 0.8 &&
                           musicEnv < 2.0;
                std::printf("  Music bass %.0fHz %.2fx -> %.4fHz "
                            "env=%.3fdB\n",
                            f0, ratio, musicHz, musicEnv);
            }
        }
    }
    std::printf("%s imbalanced low-bass f0err=%.4fHz env=%.3fdB loud=%.3fdB\n",
                bassPass ? "PASS" : "FAIL", worstBassHz, worstBassEnv,
                worstLoudDb);
    if (!bassPass) ++fails;

    // MultiRes phase/onset analysis must follow an active channel even when
    // channel 0 is digital silence. Cover both the near-unity Auto dispatcher
    // and the regular Offline-Music whole-signal path.
    const int activeN = SR * 3;
    std::vector<std::vector<float>> activeRight(
        ch, std::vector<float>(activeN, 0.0f));
    for (int i = 0; i < activeN; ++i)
        activeRight[1][i] = static_cast<float>(
            0.35 * std::sin(2.0 * kPi * 197.0 * i / SR));
    bool activeRefPass = true;
    double worstActiveHz = 0.0, worstActiveRmsDb = 0.0;
    float worstActivePeak = 0.0f, worstSilentPeak = 0.0f;
    struct ActiveRefCase {
        double ratio;
        Config::Mode mode;
    };
    for (const ActiveRefCase test : {
             ActiveRefCase{1.02, Config::Mode::Auto},
             ActiveRefCase{1.10, Config::Mode::Music}}) {
        auto y = renderAll(activeRight, 4096, test.ratio, SR,
                           Config::Tier::Offline, test.mode);
        const double hz = risingCrossingHz(y[1], SR);
        double inEnergy = 0.0, outEnergy = 0.0;
        for (float s : activeRight[1])
            inEnergy += static_cast<double>(s) * s;
        for (int c = 0; c < ch; ++c) {
            for (float s : y[c]) {
                activeRefPass = activeRefPass && std::isfinite(s);
                if (c == 0)
                    worstSilentPeak = std::max(worstSilentPeak, std::abs(s));
                else {
                    worstActivePeak = std::max(worstActivePeak, std::abs(s));
                    outEnergy += static_cast<double>(s) * s;
                }
            }
        }
        const double rmsDb = 10.0 * std::log10(
            (outEnergy / y[1].size() + 1e-30) /
            (inEnergy / activeRight[1].size() + 1e-30));
        worstActiveHz = std::max(worstActiveHz, std::abs(hz - 197.0));
        worstActiveRmsDb = std::max(worstActiveRmsDb, std::abs(rmsDb));
        activeRefPass = activeRefPass &&
                        y[0].size() == static_cast<size_t>(
                            llround(activeN * test.ratio)) &&
                        std::abs(hz - 197.0) < 0.8 &&
                        std::abs(rmsDb) < 2.0 && worstActivePeak < 1.0f &&
                        worstSilentPeak < 1e-7f;
    }
    std::printf("%s loudest-channel reference f0err=%.4fHz rms=%.3fdB "
                "activePeak=%.4f silentPeak=%.3g\n",
                activeRefPass ? "PASS" : "FAIL", worstActiveHz,
                worstActiveRmsDb, worstActivePeak, worstSilentPeak);
    if (!activeRefPass) ++fails;

    // A very quiet quadrature partner of a tonal reference is still a spatial
    // stereo pair, not an independent tonal bed. Preserve its 90-degree phase
    // and level relationship instead of classifying it by zero-lag correlation.
    std::vector<std::vector<float>> quadrature(
        ch, std::vector<float>(activeN, 0.0f));
    for (int i = 0; i < activeN; ++i) {
        const double phase = 2.0 * kPi * 197.0 * i / SR;
        quadrature[0][i] = static_cast<float>(0.35 * std::sin(phase));
        quadrature[1][i] = static_cast<float>(0.0035 * std::cos(phase));
    }
    auto quadratureY = renderAll(quadrature, 4096, 1.02, SR,
                                 Config::Tier::Offline);
    const size_t quadBegin = quadratureY[0].size() / 5;
    const size_t quadEnd = quadratureY[0].size() * 4 / 5;
    double re[2] = {0.0, 0.0}, im[2] = {0.0, 0.0};
    bool quadratureFinite = true;
    for (size_t i = quadBegin; i < quadEnd; ++i) {
        const double phase = 2.0 * kPi * 197.0 * i / SR;
        for (int c = 0; c < ch; ++c) {
            const double sample = quadratureY[c][i];
            quadratureFinite = quadratureFinite && std::isfinite(sample);
            re[c] += sample * std::cos(phase);
            im[c] -= sample * std::sin(phase);
        }
    }
    auto wrapPhase = [](double phase) {
        while (phase > kPi) phase -= 2.0 * kPi;
        while (phase < -kPi) phase += 2.0 * kPi;
        return phase;
    };
    const double quadRelative = wrapPhase(
        std::atan2(im[1], re[1]) - std::atan2(im[0], re[0]));
    const double quadPhaseError = std::abs(
        wrapPhase(quadRelative - kPi / 2.0));
    const double quadLevel = std::hypot(re[1], im[1]) /
                             std::max(1e-30, std::hypot(re[0], im[0]));
    const bool quadraturePass = quadratureFinite && quadPhaseError < 0.20 &&
                                std::abs(quadLevel - 0.01) < 0.002;
    std::printf("%s quiet quadrature stereo phaseerr=%.4frad level=%.5f\n",
                quadraturePass ? "PASS" : "FAIL", quadPhaseError, quadLevel);
    if (!quadraturePass) ++fails;

    // Time geometry scales at high sample rates, and remains deterministic.
    bool ratePass = true;
    for (int rate : {44100, 48000, 96000, 192000}) {
        const int rateN = rate / 4;
        std::vector<std::vector<float>> rateX(2,
                                              std::vector<float>(rateN));
        uint32_t state = 0xabcdef01u, stateR = 0x10293847u;
        for (int i = 0; i < rateN; ++i) {
            state = state * 1664525u + 1013904223u;
            stateR = stateR * 22695477u + 1u;
            rateX[0][i] = (static_cast<int>((state >> 8) & 0xffff) -
                           32768) * (0.25f / 32768.0f);
            rateX[1][i] = (static_cast<int>((stateR >> 8) & 0xffff) -
                           32768) * (0.0025f / 32768.0f);
        }
        auto a = renderAll(rateX, 257, 1.02, rate,
                           Config::Tier::Offline,
                           Config::Mode::Rhythm);
        auto b = renderAll(rateX, rateN, 1.02, rate,
                           Config::Tier::Offline,
                           Config::Mode::Rhythm);
        const size_t expected = static_cast<size_t>(llround(rateN * 1.02));
        ratePass = ratePass && a[0].size() == expected && b[0].size() == expected;
        for (int c = 0; c < 2 && a[c].size() == b[c].size(); ++c)
            for (size_t i = 0; i < a[c].size(); ++i)
                ratePass = ratePass && a[c][i] == b[c][i];
    }
    std::printf("%s near-unity 44.1/48/96/192 kHz geometry\n",
                ratePass ? "PASS" : "FAIL");
    if (!ratePass) ++fails;

    // Offline Music now means the actual multi-resolution renderer, while
    // exact unity still takes the literal copy path. Ratios just outside the
    // near-unity dispatch remain streaming and expose output before finish().
    auto musicUnity = renderAll(nearX, 8192, 1.0, SR,
                                Config::Tier::Offline,
                                Config::Mode::Music);
    size_t musicUnityDiffs = 0;
    for (int c = 0; c < ch; ++c)
        for (size_t i = 0; i < nearX[c].size(); ++i)
            if (musicUnity[c][i] != nearX[c][i]) ++musicUnityDiffs;
    auto musicMulti = renderAll(nearX, 8192, 1.02, SR,
                                Config::Tier::Offline,
                                Config::Mode::Music);
    Config dispatchCfg;
    dispatchCfg.sampleRate = SR;
    dispatchCfg.channels = ch;
    dispatchCfg.tier = Config::Tier::Offline;
    std::vector<const float*> dispatchIn(ch);
    for (int c = 0; c < ch; ++c) dispatchIn[c] = nearX[c].data();
    Stretcher inside, outsideLo, outsideHi;
    inside.configure(dispatchCfg);
    outsideLo.configure(dispatchCfg);
    outsideHi.configure(dispatchCfg);
    inside.setTimeStretch(1.02);
    outsideLo.setTimeStretch(0.969);
    outsideHi.setTimeStretch(1.051);
    inside.feed(dispatchIn.data(), nearN);
    outsideLo.feed(dispatchIn.data(), nearN);
    outsideHi.feed(dispatchIn.data(), nearN);
    const bool dispatchPass = musicUnityDiffs == 0 &&
                              musicMulti[0].size() ==
                                  static_cast<size_t>(llround(nearN * 1.02)) &&
                              inside.available() == 0 &&
                              outsideLo.available() > 0 &&
                              outsideHi.available() > 0;
    std::printf("%s Music-Multi/unity/dispatch unitydiff=%zu\n",
                dispatchPass ? "PASS" : "FAIL", musicUnityDiffs);
    if (!dispatchPass) ++fails;

    // Parameter changes after an Offline near-unity feed replay the buffered
    // prefix through the old regular parameters, then apply the setter.  It
    // must equal a stream that was explicitly kept on the regular path.
    auto setterRender = [&](bool forceRegular) {
        Stretcher st;
        st.configure(dispatchCfg);
        st.setTimeStretch(1.02);
        if (forceRegular) st.setFormantPreserve(true);  // pitch=0: no DSP effect
        const int split = nearN / 3;
        std::vector<const float*> ip(ch);
        for (int c = 0; c < ch; ++c) ip[c] = nearX[c].data();
        st.feed(ip.data(), split);
        st.setTimeStretch(1.06);
        long long fed = split;
        while (fed < nearN) {
            const int k = static_cast<int>(
                std::min<long long>(8192, nearN - fed));
            for (int c = 0; c < ch; ++c) ip[c] = nearX[c].data() + fed;
            st.feed(ip.data(), k);
            fed += k;
        }
        st.finish();
        std::vector<std::vector<float>> out(ch);
        std::vector<std::vector<float>> scratch(ch,
                                                std::vector<float>(8192));
        std::vector<float*> op(ch);
        for (int c = 0; c < ch; ++c) op[c] = scratch[c].data();
        while (st.available() > 0) {
            const int want = std::min(8192, st.available());
            const int made = st.read(op.data(), want);
            if (made <= 0) break;
            for (int c = 0; c < ch; ++c)
                out[c].insert(out[c].end(), scratch[c].begin(),
                              scratch[c].begin() + made);
        }
        return out;
    };
    auto setterFallback = setterRender(false);
    auto setterRegular = setterRender(true);
    size_t setterDiffs = 0;
    bool setterPass = setterFallback[0].size() == setterRegular[0].size();
    if (setterPass)
        for (int c = 0; c < ch; ++c)
            for (size_t i = 0; i < setterFallback[c].size(); ++i)
                if (setterFallback[c][i] != setterRegular[c][i])
                    ++setterDiffs;
    setterPass = setterPass && setterDiffs == 0;
    std::printf("%s in-flight setter fallback diffs=%zu\n",
                setterPass ? "PASS" : "FAIL", setterDiffs);
    if (!setterPass) ++fails;

    Stretcher nearReuse;
    nearReuse.configure(dispatchCfg);
    nearReuse.setTimeStretch(1.02);
    auto reusePassRender = [&]() {
        std::vector<const float*> ip(ch);
        for (int c = 0; c < ch; ++c) ip[c] = nearX[c].data();
        nearReuse.feed(ip.data(), nearN);
        nearReuse.finish();
        std::vector<std::vector<float>> out(
            ch, std::vector<float>(static_cast<size_t>(llround(nearN * 1.02))));
        std::vector<float*> op(ch);
        for (int c = 0; c < ch; ++c) op[c] = out[c].data();
        const int made = nearReuse.read(op.data(), static_cast<int>(out[0].size()));
        if (made != static_cast<int>(out[0].size())) out[0].clear();
        return out;
    };
    auto reuseA = reusePassRender();
    nearReuse.reset();
    auto reuseB = reusePassRender();
    size_t reuseDiffs = 0;
    bool reusePass = reuseA[0].size() == reuseB[0].size() && !reuseA[0].empty();
    if (reusePass)
        for (int c = 0; c < ch; ++c)
            for (size_t i = 0; i < reuseA[c].size(); ++i)
                if (reuseA[c][i] != reuseB[c][i]) ++reuseDiffs;
    reusePass = reusePass && reuseDiffs == 0;
    std::printf("%s near-unity reset/reuse diffs=%zu\n",
                reusePass ? "PASS" : "FAIL", reuseDiffs);
    if (!reusePass) ++fails;

    // Regression for the old 524,288-sample ring overwrite: >11 seconds in a
    // single public feed() call must equal ordinary 8192-frame streaming.
    const int longN = SR * 11 + SR / 4;
    std::vector<std::vector<float>> longX(ch, std::vector<float>(longN));
    for (int i = 0; i < longN; ++i) {
        const double t = static_cast<double>(i) / SR;
        longX[0][i] = static_cast<float>(
            0.38 * std::sin(2.0 * kPi * 223.0 * t) +
            0.12 * std::sin(2.0 * kPi * 3111.0 * t));
        longX[1][i] = static_cast<float>(
            0.31 * std::sin(2.0 * kPi * 337.0 * t + 0.3));
    }
    auto longBlock = render(longX, 8192, 1.02, 0.0, SR);
    auto longOne = render(longX, longN, 1.02, 0.0, SR);
    size_t longDiffs = 0;
    for (int c = 0; c < ch; ++c)
        for (size_t i = 0; i < longBlock[c].size(); ++i)
            if (longBlock[c][i] != longOne[c][i]) ++longDiffs;
    const bool longPass = longDiffs == 0;
    std::printf("%s oversized one-shot feed frames=%d diffs=%zu\n",
                longPass ? "PASS" : "FAIL", longN, longDiffs);
    if (!longPass) ++fails;

    // Reserve from current WOLA high-water, not absolute input*current ratio:
    // after a drained 2x epoch, a 0.5x epoch can still produce >2^19 unread
    // samples and must survive one-shot input without wraparound.
    auto ratioChangeRender = [&](int secondChunk) {
        Config cfg;
        cfg.sampleRate = 8000;
        cfg.channels = 1;
        cfg.tier = Config::Tier::StudioRT;
        Stretcher st;
        st.configure(cfg);
        st.setTimeStretch(2.0);
        const int firstN = 100000;
        const int secondN = 1100000;
        std::vector<float> first(firstN), second(secondN);
        for (int i = 0; i < firstN; ++i)
            first[i] = static_cast<float>(
                0.3 * std::sin(2.0 * kPi * 173.0 * i / cfg.sampleRate));
        for (int i = 0; i < secondN; ++i)
            second[i] = static_cast<float>(
                0.25 * std::sin(2.0 * kPi * 227.0 * i / cfg.sampleRate));
        std::vector<float> out;
        std::vector<float> scratch(16384);
        float* op[] = {scratch.data()};
        auto drain = [&]() {
            while (st.available() > 0) {
                const int want = std::min(16384, st.available());
                const int made = st.read(op, want);
                if (made <= 0) break;
                out.insert(out.end(), scratch.begin(), scratch.begin() + made);
            }
        };
        const float* p1[] = {first.data()};
        st.feed(p1, firstN);
        drain();
        st.setTimeStretch(0.5);
        int fed = 0;
        while (fed < secondN) {
            const int k = std::min(secondChunk, secondN - fed);
            const float* p2[] = {second.data() + fed};
            st.feed(p2, k);
            fed += k;
            if (secondChunk < secondN) drain();
        }
        drain();
        st.finish();
        drain();
        return out;
    };
    auto ratioLarge = ratioChangeRender(1100000);
    auto ratioBlocks = ratioChangeRender(8192);
    size_t ratioDiffs = 0;
    const bool ratioLengths = ratioLarge.size() == ratioBlocks.size();
    if (ratioLengths)
        for (size_t i = 0; i < ratioLarge.size(); ++i)
            if (ratioLarge[i] != ratioBlocks[i]) ++ratioDiffs;
    const bool ratioChangePass = ratioLengths && ratioDiffs == 0;
    std::printf("%s high-to-low ratio oversized feed samples=%zu diffs=%zu\n",
                ratioChangePass ? "PASS" : "FAIL", ratioLarge.size(),
                ratioDiffs);
    if (!ratioChangePass) ++fails;

    std::printf(fails ? "\n%d FAILURES\n" : "\nALL PASS\n", fails);
    return fails ? 1 : 0;
}
