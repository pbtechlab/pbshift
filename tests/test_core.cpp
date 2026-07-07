// Core correctness gates (M1):
//  1. WOLA identity reconstruction < -120 dB (the bypass-null CI gate)
//  2. reassignment omega accuracy on a pure sine
//  3. reassignment tau accuracy on an offset impulse
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

#include "pb_fft.h"
#include "pb_stft.h"
#include "pb_window.h"

using namespace pbshift;

static int fails = 0;
#define CHECK(cond, msg, val)                                        \
    do {                                                             \
        if (cond) {                                                  \
            std::printf("PASS  %-40s %.6g\n", msg, (double)(val));   \
        } else {                                                     \
            std::printf("FAIL  %-40s %.6g\n", msg, (double)(val));   \
            ++fails;                                                 \
        }                                                            \
    } while (0)

int main() {
    const int N = 4096, hop = N / 4;
    const int SR = 48000;
    WindowSet win = WindowSet::hann(N);
    AnalysisFrontend fe(N, win);
    WolaSynth ws(N, win, 1 << 18);

    // --- 1. identity reconstruction ---------------------------------
    const int LEN = SR;  // 1 s
    std::vector<float> x(LEN + 2 * N, 0.0f);
    for (int i = 0; i < LEN + 2 * N; ++i)
        x[i] = 0.5f * std::sin(2.0 * M_PI * 440.0 * i / SR) +
               0.3f * std::sin(2.0 * M_PI * 1237.0 * i / SR + 0.7) +
               0.1f * std::sin(2.0 * M_PI * 7000.0 * i / SR + 2.1);

    AnalysisFrame fr;
    std::vector<float> frame(N);
    for (long long c = 0; c * hop < LEN + N; ++c) {
        const long long center = c * hop;
        for (int i = 0; i < N; ++i) {
            const long long p = center - N / 2 + i;
            frame[i] = (p >= 0 && p < (long long)x.size()) ? x[p] : 0.0f;
        }
        fe.analyze(frame.data(), fr);
        ws.addFrame(fr.X.data(), center);
    }
    std::vector<float> y(LEN);
    ws.read(0, LEN, y.data());
    double err = 0, ref = 0;
    for (int i = N; i < LEN - N; ++i) {  // skip edges
        const double d = y[i] - x[i];
        err += d * d;
        ref += (double)x[i] * x[i];
    }
    const double nullDb = 10.0 * std::log10(err / ref + 1e-300);
    CHECK(nullDb < -120.0, "WOLA identity null (dB)", nullDb);

    // --- 2. omega accuracy on pure sine ------------------------------
    const double f0 = 997.0;
    for (int i = 0; i < N; ++i)
        frame[i] = std::sin(2.0 * M_PI * f0 * i / SR + 0.3);
    fe.analyze(frame.data(), fr);
    const double binHz = (double)SR / N;
    const int peak = (int)std::round(f0 / binHz);
    double maxCentsErr = 0;
    for (int m = peak - 1; m <= peak + 1; ++m) {
        const double fHat = fr.omega[m] * SR / (2.0 * M_PI);
        const double cents = 1200.0 * std::log2(fHat / f0);
        if (std::abs(cents) > maxCentsErr) maxCentsErr = std::abs(cents);
    }
    CHECK(maxCentsErr < 0.5, "IF accuracy near peak (cents)", maxCentsErr);

    // --- 3. tau accuracy on offset impulse ---------------------------
    const int off = 300;  // samples after window center
    std::fill(frame.begin(), frame.end(), 0.0f);
    frame[N / 2 + off] = 1.0f;
    fe.analyze(frame.data(), fr);
    double tauErr = 0;
    int cnt = 0;
    for (int m = 40; m < N / 2 - 40; m += 16) {
        if (fr.mag[m] < 1e-6f) continue;
        tauErr += std::abs(fr.tau[m] - off);
        ++cnt;
    }
    tauErr /= cnt > 0 ? cnt : 1;
    CHECK(tauErr < 1.0, "group-delay accuracy (samples)", tauErr);

    std::printf(fails ? "\n%d FAILURES\n" : "\nALL PASS\n", fails);
    return fails ? 1 : 0;
}
