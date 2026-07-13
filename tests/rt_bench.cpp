// Real-time viability: stream stereo audio in small blocks and measure
// per-block processing cost against the real-time budget.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "pbshift/pbshift.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(int argc, char** argv) {
    const int SR = 48000, ch = 2, BLOCK = argc > 1 ? std::atoi(argv[1]) : 512;
    const double stretch = argc > 2 ? std::atof(argv[2]) : 1.0;
    const double pitch = argc > 3 ? std::atof(argv[3]) : 7.0;
    const char* mode = argc > 4 ? argv[4] : "auto";
    const int seconds = 30;

    pbshift::Config cfg;
    cfg.sampleRate = SR;
    cfg.channels = ch;
    if (!std::strcmp(mode, "music"))
        cfg.mode = pbshift::Config::Mode::Music;
    else if (!std::strcmp(mode, "rhythm"))
        cfg.mode = pbshift::Config::Mode::Rhythm;
    else if (!std::strcmp(mode, "voice"))
        cfg.mode = pbshift::Config::Mode::Voice;
    pbshift::Stretcher st;
    st.configure(cfg);
    st.setTimeStretch(stretch);
    st.setPitchSemitones(pitch);
    st.setFormantPreserve(true);

    std::vector<std::vector<float>> in(ch, std::vector<float>(BLOCK));
    std::vector<std::vector<float>> out(ch, std::vector<float>(4 * BLOCK + 64));
    std::vector<const float*> ip(ch);
    std::vector<float*> op(ch);
    for (int c = 0; c < ch; ++c) {
        ip[c] = in[c].data();
        op[c] = out[c].data();
    }

    const int blocks = seconds * SR / BLOCK;
    double worst = 0.0, total = 0.0;
    long long phase = 0;
    for (int b = 0; b < blocks; ++b) {
        for (int c = 0; c < ch; ++c)
            for (int i = 0; i < BLOCK; ++i) {
                const double t = static_cast<double>(phase + i) / SR;
                in[c][i] = static_cast<float>(
                    0.4 * std::sin(2 * M_PI * (220 + 30 * c) * t) +
                    0.2 * std::sin(2 * M_PI * 1730 * t) +
                    ((phase + i) % 12000 == 0 ? 0.7 : 0.0));
            }
        phase += BLOCK;
        const auto t0 = std::chrono::steady_clock::now();
        st.feed(ip.data(), BLOCK);
        const int want = static_cast<int>(BLOCK * stretch);
        if (st.available() >= want) st.read(op.data(), want);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (b > 4 && ms > worst) worst = ms;  // skip warmup
        total += ms;
    }
    const double budget = 1000.0 * BLOCK / SR;
    std::printf("block=%d mode=%s stretch=%.2f pitch=%+.0f  "
                "avg=%.3fms worst=%.3fms "
                "budget=%.2fms  worst-load=%.1f%%  latency in/out=%d/%d\n",
                BLOCK, mode, stretch, pitch, total / blocks, worst, budget,
                100.0 * worst / budget, st.inputLatency(), st.outputLatency());
    return worst < budget ? 0 : 1;
}
