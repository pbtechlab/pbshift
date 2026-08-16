// Gates for the pitch-synchronous Voice engine.
//
// The engine is streaming and causal, and every one of these checks exists
// because the corresponding property was broken at some point during its
// development: the schedule anchored on whatever the tracker had reached, a
// grain's search saw more input the longer it took, the ratio was silently
// clamped narrower than the engine's own range, and a mid-stream ratio change
// teleported the read position past the end of the audio.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "pbshift/pbshift.h"

using namespace pbshift;

namespace {

int failures = 0;

void check(bool ok, const char* what, const char* detail = "") {
    std::printf("%s %s %s\n", ok ? "PASS" : "FAIL", what, detail);
    if (!ok) ++failures;
}

std::vector<std::vector<float>> voiceSignal(int sr, int n, int ch) {
    // A buzzy periodic source with a breathy top: voiced enough for the tracker
    // to lock, broadband enough to exercise the crossover.
    std::vector<std::vector<float>> x(ch, std::vector<float>(n, 0.0f));
    const double f0 = 120.0;
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / sr;
        double v = 0.0;
        for (int h = 1; h <= 12; ++h)
            v += std::sin(2.0 * M_PI * f0 * h * t) / h;
        v *= 0.25;
        v += 0.02 * std::sin(2.0 * M_PI * 6100.0 * t);
        for (int c = 0; c < ch; ++c)
            x[c][i] = static_cast<float>(v * (c == 0 ? 1.0 : 0.8));
    }
    return x;
}

struct Render {
    std::vector<std::vector<float>> y;
    int inputLatency = 0, outputLatency = 0;
};

Render render(const std::vector<std::vector<float>>& x, int chunk,
              double stretch, double pitch, int sr,
              double changeStretchAt = -1.0, double newStretch = 1.0) {
    const int ch = static_cast<int>(x.size());
    Config cfg;
    cfg.sampleRate = sr;
    cfg.channels = ch;
    cfg.mode = Config::Mode::Voice;
    Stretcher st;
    st.configure(cfg);
    st.setTimeStretch(stretch);
    st.setPitchSemitones(pitch);

    const long long n = static_cast<long long>(x[0].size());
    Render out;
    out.y.assign(ch, {});
    std::vector<const float*> ip(ch);
    std::vector<float*> op(ch);
    std::vector<std::vector<float>> buf(ch, std::vector<float>(1 << 16));
    for (int c = 0; c < ch; ++c) op[c] = buf[c].data();

    auto drain = [&]() {
        for (;;) {
            const int avail = st.available();
            if (avail <= 0) break;
            const int want = std::min(avail, 1 << 16);
            const int got = st.read(op.data(), want);
            if (got <= 0) break;
            for (int c = 0; c < ch; ++c)
                out.y[c].insert(out.y[c].end(), buf[c].begin(),
                                buf[c].begin() + got);
        }
    };

    long long fed = 0;
    bool changed = false;
    while (fed < n) {
        const int k = static_cast<int>(std::min<long long>(chunk, n - fed));
        for (int c = 0; c < ch; ++c) ip[c] = x[c].data() + fed;
        st.feed(ip.data(), k);
        fed += k;
        if (!changed && changeStretchAt > 0.0 &&
            fed >= static_cast<long long>(changeStretchAt * n)) {
            st.setTimeStretch(newStretch);
            changed = true;
        }
        drain();
    }
    st.finish();
    drain();
    out.inputLatency = st.inputLatency();
    out.outputLatency = st.outputLatency();
    return out;
}

size_t diffCount(const std::vector<std::vector<float>>& a,
                 const std::vector<std::vector<float>>& b) {
    size_t d = 0;
    for (size_t c = 0; c < a.size(); ++c) {
        const size_t n = std::min(a[c].size(), b[c].size());
        for (size_t i = 0; i < n; ++i)
            if (a[c][i] != b[c][i]) ++d;
        d += std::max(a[c].size(), b[c].size()) - n;
    }
    return d;
}

}  // namespace

int main() {
    const int SR = 48000, ch = 2;
    const auto x = voiceSignal(SR, SR * 2, ch);

    // Host buffer size must not reach the output. 1-sample feeds and one-shot
    // feeds have to agree with the ordinary block sizes exactly.
    for (double ratio : {0.25, 0.5, 1.25, 2.0, 4.0}) {
        const auto base = render(x, 512, ratio, 0.0, SR);
        char msg[128];
        bool ok = true;
        for (int chunk : {1, 37, 64, 8192, 1 << 20}) {
            const auto other = render(x, chunk, ratio, 0.0, SR);
            const size_t d = diffCount(base.y, other.y);
            if (d != 0) {
                std::snprintf(msg, sizeof msg, "ratio=%.2f chunk=%d diffs=%zu",
                              ratio, chunk, d);
                ok = false;
                break;
            }
        }
        if (ok)
            std::snprintf(msg, sizeof msg, "ratio=%.2f", ratio);
        check(ok, "voice chunk-independence", msg);
    }

    // Length is exact, including ratios outside the range the engine was tuned
    // on: a silent internal clamp there would put the whole stream off the
    // host's timeline, not just its tail.
    for (double ratio : {0.1, 0.25, 1.0, 4.0, 8.0}) {
        const auto r = render(x, 512, ratio, 0.0, SR);
        const long long want = llround(x[0].size() * ratio);
        char msg[128];
        std::snprintf(msg, sizeof msg, "ratio=%.2f got=%zu want=%lld", ratio,
                      r.y[0].size(), want);
        check(static_cast<long long>(r.y[0].size()) == want,
              "voice exact length", msg);
    }

    // A ratio change mid-stream must bend the schedule from that point, not
    // re-map the past: the output has to keep flowing.
    {
        const auto r = render(x, 512, 1.0, 0.0, SR, 0.5, 0.5);
        char msg[128];
        std::snprintf(msg, sizeof msg, "produced=%zu", r.y[0].size());
        check(r.y[0].size() > x[0].size() / 2, "voice mid-stream ratio change",
              msg);
    }

    // Degenerate inputs must not crash or hang.
    {
        std::vector<std::vector<float>> tiny(ch, std::vector<float>(100, 0.1f));
        const auto r = render(tiny, 16, 2.0, 0.0, SR);
        check(r.y[0].size() == 200, "voice very short input",
              r.y[0].size() == 200 ? "" : "length mismatch");

        std::vector<std::vector<float>> silence(ch, std::vector<float>(SR, 0.0f));
        const auto s = render(silence, 512, 2.0, 0.0, SR);
        bool quiet = true;
        for (int c = 0; c < ch; ++c)
            for (float v : s.y[c])
                if (std::fabs(v) > 1e-6f) quiet = false;
        check(s.y[0].size() == static_cast<size_t>(2 * SR) && quiet,
              "voice silence stays silent");

        std::vector<std::vector<float>> none(ch, std::vector<float>());
        const auto e = render(none, 512, 2.0, 0.0, SR);
        check(e.y[0].empty(), "voice empty stream");
    }

    // The pitch floor must not rise with the sample rate: the analysis frame
    // has to grow so the longest period still fits inside it twice.
    for (int sr : {44100, 96000}) {
        const auto sig = voiceSignal(sr, sr, ch);
        const auto a = render(sig, 512, 2.0, 0.0, sr);
        const auto b = render(sig, 64, 2.0, 0.0, sr);
        char msg[128];
        std::snprintf(msg, sizeof msg, "sr=%d latency=%d/%d diffs=%zu", sr,
                      a.inputLatency, a.outputLatency, diffCount(a.y, b.y));
        check(diffCount(a.y, b.y) == 0 &&
                  a.y[0].size() == static_cast<size_t>(2 * sr),
              "voice sample-rate handling", msg);
    }

    std::printf("\n%d FAILURES\n", failures);
    return failures == 0 ? 0 : 1;
}
