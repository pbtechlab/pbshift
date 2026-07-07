// Example: driving pbshift as a real-time insert with a FIXED host block
// size (the shape a VST3/AU processBlock would take). The pull API decouples
// input feed from output demand, so a stretch != 1 does not need a matching
// number of output samples per call — you read whatever is available.
//
// For a pitch-shift-only insert (stretch == 1, the common plugin case) the
// engine is latency-compensated: report inputLatency()+outputLatency() to the
// host for PDC, prime that many output samples once, then it runs 1:1.
//
// Build (from repo root, after the library is built):
//   g++ -O2 -std=c++17 -Iinclude examples/realtime_insert.cpp \
//       -Lbuild -lpbshift -lpffft_static -o realtime_insert   (or add to CMake)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "pbshift/pbshift.h"

int main() {
    const int SR = 48000, CH = 2, BLOCK = 256;  // typical DAW block
    pbshift::Config cfg;
    cfg.sampleRate = SR;
    cfg.channels = CH;

    pbshift::Stretcher st;
    st.configure(cfg);
    st.setPitchSemitones(7.0);   // +7 st insert, no time change
    st.setTimeStretch(1.0);
    st.setFormantPreserve(true);

    const int latency = st.inputLatency() + st.outputLatency();
    std::printf("report %d samples of latency to the host for PDC\n", latency);

    std::vector<std::vector<float>> in(CH, std::vector<float>(BLOCK));
    std::vector<std::vector<float>> out(CH, std::vector<float>(BLOCK));
    std::vector<const float*> ip(CH);
    std::vector<float*> op(CH);
    for (int c = 0; c < CH; ++c) {
        ip[c] = in[c].data();
        op[c] = out[c].data();
    }

    // Prime: feed silence until `latency` output samples are queued, so that
    // from the host's view output[n] corresponds to input[n] (minus PDC).
    std::vector<std::vector<float>> zero(CH, std::vector<float>(BLOCK, 0.0f));
    std::vector<const float*> zp(CH);
    for (int c = 0; c < CH; ++c) zp[c] = zero[c].data();
    int primed = 0;
    while (st.available() < latency) {
        st.feed(zp.data(), BLOCK);
        primed += BLOCK;
    }
    std::vector<float> sink(BLOCK);
    std::vector<float*> skp(CH);
    for (int c = 0; c < CH; ++c) skp[c] = sink.data();
    int drop = latency;
    while (drop > 0) {  // discard the priming latency
        const int k = std::min(drop, BLOCK);
        st.read(skp.data(), k);
        drop -= k;
    }

    // Steady state: exactly BLOCK in, BLOCK out per callback.
    long long phase = 0;
    double peak = 0.0;
    for (int b = 0; b < SR / BLOCK; ++b) {  // 1 second
        for (int c = 0; c < CH; ++c)
            for (int i = 0; i < BLOCK; ++i)
                in[c][i] = 0.4f * std::sin(2 * 3.14159265 * 220 * (phase + i) / SR);
        phase += BLOCK;
        st.feed(ip.data(), BLOCK);
        const int got = st.read(op.data(), BLOCK);
        if (got != BLOCK) {
            std::printf("underrun at block %d (got %d)\n", b, got);
            return 1;
        }
        for (int i = 0; i < BLOCK; ++i)
            peak = std::max(peak, static_cast<double>(std::abs(op[0][i])));
    }
    std::printf("steady-state 1:1 insert OK, output peak %.3f\n", peak);
    return 0;
}
