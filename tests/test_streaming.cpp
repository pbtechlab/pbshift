// Streaming determinism gates:
//  1. chunk-size independence: feeding 64 vs 8192 samples per call must
//     produce bit-identical output (host-buffer independence)
//  2. re-render determinism: two runs, identical bits
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "pbshift/pbshift.h"

using pbshift::Config;
using pbshift::Stretcher;

static std::vector<std::vector<float>> render(
    const std::vector<std::vector<float>>& x, int chunk,
    double stretch, double pitch, int sr,
    Config::Mode mode = Config::Mode::Auto) {
    const int ch = static_cast<int>(x.size());
    Config cfg;
    cfg.sampleRate = sr;
    cfg.channels = ch;
    cfg.mode = mode;
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
    std::printf(fails ? "\n%d FAILURES\n" : "\nALL PASS\n", fails);
    return fails ? 1 : 0;
}
