#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "pb_speech_pitch.h"

// A weak periodic voice must not be treated as an entirely unpitched frame.
// Exercise the speech backend directly, independently of Auto's content router.
using Audio = std::vector<std::vector<float>>;
static Audio render(const Audio &input, int sr, int block, int readBlock) {
    pbshift::SpeechPitchProcessor engine;
    engine.configure(sr, 2, 2048);
    engine.setPitch(24);
    int count = int(input[0].size()), written = 0;
    Audio output(2, std::vector<float>(count));
    const float *ip[2]; float *op[2];
    auto drain = [&]() {
        while (engine.available()) {
            int n = std::min({readBlock, engine.available(), count - written});
            if (n <= 0) break;
            for (int c = 0; c < 2; ++c) op[c] = output[c].data() + written;
            int got = engine.read(op, n);
            if (got != n) std::abort();
            written += got;
        }
    };
    for (int at = 0; at < count;) {
        int n = std::min(block, count - at);
        for (int c = 0; c < 2; ++c) ip[c] = input[c].data() + at;
        engine.feed(ip, n); at += n; drain();
    }
    engine.finish(count); drain();
    if (written != count) std::abort();
    return output;
}
static double linePower(const std::vector<float> &x, int sr, double frequency) {
    constexpr double pi = 3.14159265358979323846;
    std::complex<double> sum = 0;
    for (int i = 0; i < sr; ++i) {
        double w = .5 - .5 * std::cos(2 * pi * i / (sr - 1));
        sum += double(x[sr + i]) * w * std::polar(1.0, -2 * pi * frequency * i / sr);
    }
    return std::norm(sum);
}
int main() {
    constexpr double pi = 3.14159265358979323846, f0 = 160;
    int failures = 0;
    // Reproduction of the weak-voice bypass defect. Broader-rate quality is
    // assessed separately; this is not a universal absence-of-residue test.
    for (int sr : {16000}) {
        std::mt19937 random(704192);
        std::normal_distribution<float> noise(0, 1);
        Audio input(2, std::vector<float>(3 * sr));
        for (int i = 0; i < 3 * sr; ++i) {
            double phase = 2 * pi * f0 * i / sr;
            input[0][i] = float(.12 * std::sin(phase) + .04 * std::sin(2 * phase)
                + .02 * std::sin(3 * phase) + .025 * noise(random));
        }
        auto a = render(input, sr, 8192, 8192), b = render(input, sr, 1, 37);
        double original = 0, wanted = 0, peak = 0; bool finite = true;
        for (int k : {1, 2, 3}) original += linePower(a[0], sr, f0 * k);
        for (int k : {4, 8, 12}) wanted += linePower(a[0], sr, f0 * k);
        for (auto &channel : a) for (float value : channel) {
            finite &= std::isfinite(value); peak = std::max(peak, std::abs(double(value)));
        }
        bool silent = std::all_of(a[1].begin(), a[1].end(), [](float v) { return v == 0; });
        bool pass = a == b && finite && silent && peak <= .99901 && wanted > 100 * original;
        std::printf("speech pitch residue sr=%d original/target=%.6g pass=%d\n",
            sr, original / std::max(1e-30, wanted), pass);
        failures += !pass;
    }
    return failures ? 1 : 0;
}
