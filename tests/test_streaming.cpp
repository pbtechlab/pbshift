// Streaming determinism gates:
//  1. chunk-size independence: feeding 64 vs 8192 samples per call must
//     produce bit-identical output (host-buffer independence)
//  2. re-render determinism: two runs, identical bits
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "pbshift/pbshift.h"

using pbshift::Config;
using pbshift::Stretcher;

static std::vector<std::vector<float>> render(
    const std::vector<std::vector<float>>& x, int chunk,
    double stretch, double pitch, int sr) {
    const int ch = static_cast<int>(x.size());
    Config cfg;
    cfg.sampleRate = sr;
    cfg.channels = ch;
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

    int fails = 0;
    for (const auto& cond : {std::pair<double, double>{1.7, 0.0},
                             {0.6, 0.0},
                             {1.0, 7.0},
                             {1.0, -7.0}}) {
        auto a = render(x, 64, cond.first, cond.second, SR);
        auto b = render(x, 8192, cond.first, cond.second, SR);
        size_t diff = 0;
        for (int c = 0; c < ch; ++c)
            for (size_t i = 0; i < a[c].size(); ++i)
                if (a[c][i] != b[c][i]) ++diff;
        auto a2 = render(x, 64, cond.first, cond.second, SR);
        size_t diff2 = 0;
        for (int c = 0; c < ch; ++c)
            for (size_t i = 0; i < a[c].size(); ++i)
                if (a[c][i] != a2[c][i]) ++diff2;
        std::printf("%s stretch=%.2f pitch=%+.0f chunk-independence diffs=%zu "
                    "rerender diffs=%zu\n",
                    (diff == 0 && diff2 == 0) ? "PASS" : "FAIL",
                    cond.first, cond.second, diff, diff2);
        if (diff || diff2) ++fails;
    }
    std::printf(fails ? "\n%d FAILURES\n" : "\nALL PASS\n", fails);
    return fails ? 1 : 0;
}
