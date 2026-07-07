// Analysis/synthesis windows and their derivative / time-weighted variants
// (needed for reassignment-operator phase gradients).
#pragma once
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace pbshift {

// Periodic Hann of length n: w[i] = 0.5 - 0.5 cos(2*pi*i/n).
// Analytic derivative dw/di = (pi/n) sin(2*pi*i/n).
struct WindowSet {
    std::vector<float> w;    // window
    std::vector<float> dw;   // d w / d sample
    std::vector<float> tw;   // (i - center) * w[i]
    double center = 0.0;     // center index (n/2 for periodic Hann)

    static WindowSet hann(int n) {
        WindowSet ws;
        ws.w.resize(n);
        ws.dw.resize(n);
        ws.tw.resize(n);
        ws.center = n * 0.5;
        const double k = 2.0 * M_PI / n;
        for (int i = 0; i < n; ++i) {
            ws.w[i] = static_cast<float>(0.5 - 0.5 * std::cos(k * i));
            ws.dw[i] = static_cast<float>(0.5 * k * std::sin(k * i));
            ws.tw[i] = static_cast<float>((i - ws.center) * ws.w[i]);
        }
        return ws;
    }
};

}  // namespace pbshift
