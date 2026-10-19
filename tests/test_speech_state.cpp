#include <cmath>
#include <cstdio>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "pb_speech_time.h"
int main() {
    pbshift::SpeechTimeProcessor processor;
    processor.configure(48000, 1, 2048);
    processor.setStretch(1);
    std::vector<float> input(256), output(8192);
    const float* ip[] = {input.data()};
    float* op[] = {output.data()};
    double phase = 0;
    for (int offset = 0; offset < 48 * 48000; offset += 256) {
        for (int i = 0; i < 256; ++i) {
            const double hz = offset + i < 48000 ? 220 : 330;
            phase += 2 * 3.14159265358979323846 * hz / 48000;
            input[i] = float(.3 * std::sin(phase));
        }
        processor.feed(ip, 256);
        while (processor.available()) processor.read(op, 8192);
    }
    const double old = processor.inputFundamental(24000);
    const double recent = processor.inputFundamental(46 * 48000);
    std::printf("retained old=%.4f recent=%.4f\n", old, recent);
    if (std::abs(old - 220) > 2 || std::abs(recent - 330) > 2) return 1;
    processor.releaseFundamentalBefore(44 * 48000);
    if (processor.inputFundamental(24000) != 0 || std::abs(processor.inputFundamental(46 * 48000) - 330) > 2) return 2;
    processor.reset();
    if (processor.inputFundamental(24000) != 0) return 3;
    return 0;
}
