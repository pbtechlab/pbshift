// Burg linear-prediction extrapolation, used to give stream edges plausible
// "virtual" neighbours (instead of zeros) so that the first/last analysis and
// resampling windows of a steady signal see a steady signal.
//
// Deterministic pure function of its input (double arithmetic, fixed loop
// order), so it preserves bit-identical output for any feed() chunking as
// long as the caller trains on a fixed span.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace pbshift {

// Predict `ext` samples that follow x[0..n). x is the training span in the
// direction of extrapolation (for backward extrapolation pass the reversed
// span). out[k] is the (k+1)-th predicted sample. Silence or too-short input
// yields zeros.
inline void burgExtrapolateRaw(const float* x, int n, int order, int ext,
                            float* out) {
    for (int k = 0; k < ext; ++k) out[k] = 0.0f;
    order = std::min(order, n / 4);
    if (order < 2 || n < 16) return;
    double energy = 0.0;
    for (int i = 0; i < n; ++i) energy += static_cast<double>(x[i]) * x[i];
    if (energy < 1e-20) return;

    std::vector<double> ef(x, x + n), eb(x, x + n);
    std::vector<double> a(order + 1, 0.0), t(order + 1, 0.0);
    a[0] = 1.0;
    int used = 0;
    // tiny regularisation keeps pure tones (den -> 0) well defined
    const double reg = 1e-12 * energy / n;
    for (int m = 0; m < order; ++m) {
        double num = 0.0, den = 0.0;
        for (int i = m + 1; i < n; ++i) {
            num += ef[i] * eb[i - 1];
            den += ef[i] * ef[i] + eb[i - 1] * eb[i - 1];
        }
        den += reg;
        if (den <= 0.0) break;
        double k = -2.0 * num / den;
        k = std::max(-0.999999, std::min(0.999999, k));
        for (int i = 1; i <= m; ++i) t[i] = a[i] + k * a[m + 1 - i];
        for (int i = 1; i <= m; ++i) a[i] = t[i];
        a[m + 1] = k;
        used = m + 1;
        for (int i = n - 1; i >= m + 1; --i) {
            const double f = ef[i];
            ef[i] = f + k * eb[i - 1];
            eb[i] = eb[i - 1] + k * f;
        }
    }
    if (used < 1) return;
    // recursive prediction over a rolling history
    std::vector<double> h(static_cast<size_t>(used) + static_cast<size_t>(ext));
    for (int i = 0; i < used; ++i) h[i] = x[n - used + i];
    for (int k = 0; k < ext; ++k) {
        double p = 0.0;
        const int base = used + k;
        for (int i = 1; i <= used; ++i) p -= a[i] * h[base - i];
        h[base] = p;
        out[k] = static_cast<float>(p);
    }
}

// Forward-backward least-squares ("modified covariance") predictor. Far more
// accurate than Burg on near-pure tones (Burg's frequency bias gives only
// ~-55 dB over a few hundred samples; this gives ~-150 dB), but not guaranteed
// stable, so the caller validates the extrapolation. Returns false when the
// normal equations are not solvable.
inline bool covariancePredictor(const float* x, int n, int p,
                                std::vector<double>& a) {
    const int q = p + 1;
    std::vector<double> R(static_cast<size_t>(q) * q, 0.0);
    // C(i,j) = sum_{t=p}^{n-1} x[t-i] x[t-j]   (forward)
    // B(i,j) = sum_{u=0}^{n-1-p} x[u+i] x[u+j] (backward)
    // first rows by direct sums, the rest by diagonal recursion
    std::vector<double> C(static_cast<size_t>(q) * q, 0.0), B = C;
    for (int j = 0; j < q; ++j) {
        double c = 0.0, b = 0.0;
        for (int t = p; t < n; ++t) c += static_cast<double>(x[t]) * x[t - j];
        for (int u = 0; u <= n - 1 - p; ++u)
            b += static_cast<double>(x[u]) * x[u + j];
        C[j] = c;
        B[j] = b;
    }
    for (int i = 0; i + 1 < q; ++i)
        for (int j = i; j + 1 < q; ++j) {
            // symmetric: fill (i+1, j+1) from (i, j); for i==0 the first row
            // (0,j) is the direct sum above
            const double cv = C[static_cast<size_t>(i) * q + j] +
                              static_cast<double>(x[p - 1 - i]) * x[p - 1 - j] -
                              static_cast<double>(x[n - 1 - i]) * x[n - 1 - j];
            C[static_cast<size_t>(i + 1) * q + j + 1] = cv;
            const double bv = B[static_cast<size_t>(i) * q + j] -
                              static_cast<double>(x[i]) * x[j] +
                              static_cast<double>(x[n - p + i]) * x[n - p + j];
            B[static_cast<size_t>(i + 1) * q + j + 1] = bv;
        }
    for (int i = 0; i < q; ++i)
        for (int j = i; j < q; ++j) {
            const double v = C[static_cast<size_t>(i) * q + j] +
                             B[static_cast<size_t>(i) * q + j];
            R[static_cast<size_t>(i) * q + j] = v;
            R[static_cast<size_t>(j) * q + i] = v;
        }
    double tr = 0.0;
    for (int i = 0; i < q; ++i) tr += R[static_cast<size_t>(i) * q + i];
    if (!(tr > 1e-20)) return false;
    // normal equations A a = b on the p x p lower block, ridge-regularised
    std::vector<double> A(static_cast<size_t>(p) * p), rhs(p);
    const double ridge = 1e-12 * tr / q;
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j)
            A[static_cast<size_t>(i) * p + j] =
                R[static_cast<size_t>(i + 1) * q + j + 1] + (i == j ? ridge : 0.0);
        rhs[i] = -R[static_cast<size_t>(i + 1) * q];
    }
    // Cholesky A = L L^T
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[static_cast<size_t>(i) * p + j];
            for (int k = 0; k < j; ++k)
                sum -= A[static_cast<size_t>(i) * p + k] * A[static_cast<size_t>(j) * p + k];
            if (i == j) {
                if (!(sum > 0.0)) return false;
                A[static_cast<size_t>(i) * p + i] = std::sqrt(sum);
            } else {
                A[static_cast<size_t>(i) * p + j] = sum / A[static_cast<size_t>(j) * p + j];
            }
        }
    }
    std::vector<double> y(p);
    for (int i = 0; i < p; ++i) {
        double sum = rhs[i];
        for (int k = 0; k < i; ++k) sum -= A[static_cast<size_t>(i) * p + k] * y[k];
        y[i] = sum / A[static_cast<size_t>(i) * p + i];
    }
    a.assign(p + 1, 0.0);
    a[0] = 1.0;
    for (int i = p - 1; i >= 0; --i) {
        double sum = y[i];
        for (int k = i + 1; k < p; ++k) sum -= A[static_cast<size_t>(k) * p + i] * a[k + 1];
        a[i + 1] = sum / A[static_cast<size_t>(i) * p + i];
    }
    return true;
}

namespace detail {
// Extrapolate `ext` samples with predictor a (order p) from the last p
// samples of x[0..n); returns false if it grows beyond `limit`.
inline bool runPredictor(const float* x, int n, const std::vector<double>& a,
                         int p, int ext, double limit, std::vector<double>& h) {
    h.assign(static_cast<size_t>(p) + static_cast<size_t>(ext), 0.0);
    for (int i = 0; i < p; ++i) h[i] = x[n - p + i];
    for (int k = 0; k < ext; ++k) {
        double v = 0.0;
        const int base = p + k;
        for (int i = 1; i <= p; ++i) v -= a[i] * h[base - i];
        h[base] = v;
        if (!(std::abs(v) <= limit)) return false;  // also catches NaN
    }
    return true;
}
}  // namespace detail

namespace detail {
// Continue x[0..n) by `count` samples as a periodic signal of (fractional)
// period P: y[n+k] = y[n+k-P], the fractional delay by a Hann-windowed sinc
// (16 taps each side). The delay is the same for every sample, so the 32
// normalised taps are computed once.
inline void runPeriodic(const float* x, int n, double P, int count,
                        std::vector<double>& y) {
    const int L = 16;
    y.assign(static_cast<size_t>(n) + static_cast<size_t>(count), 0.0);
    for (int i = 0; i < n; ++i) y[static_cast<size_t>(i)] = x[i];
    const double p0 = static_cast<double>(n) - P;
    const long i0 = static_cast<long>(std::floor(p0));
    const double fr = p0 - static_cast<double>(i0);
    double h[2 * L], wsum = 0.0;
    for (int j = -L + 1; j <= L; ++j) {
        const double d = static_cast<double>(j) - fr;
        const double s = std::abs(d) < 1e-12 ? 1.0
                         : std::sin(3.14159265358979 * d) / (3.14159265358979 * d);
        const double w = 0.5 + 0.5 * std::cos(3.14159265358979 * d / (L + 1));
        h[j + L - 1] = s * w;
        wsum += s * w;
    }
    if (wsum != 0.0)
        for (double& v : h) v /= wsum;
    for (int k = 0; k < count; ++k) {
        const long b = i0 + k;
        double acc = 0.0;
        for (int j = -L + 1; j <= L; ++j)
            acc += y[static_cast<size_t>(b + j)] * h[j + L - 1];
        y[static_cast<size_t>(n + k)] = acc;
    }
}
}  // namespace detail

// Periodic continuation of a strongly periodic edge (a sustained pitched
// tone). An AR predictor of modest order cannot carry a tone with more
// partials than half its order (a 110 Hz sawtooth has 181 below 20 kHz, the
// order-48 predictor models 24): its extension detunes and drops them, so
// the frames that straddle the end of the stream see a dulled tone and the
// output's last milliseconds sag (A2 sawtooth +12 st: the closing attack
// read 10.5 dB weaker than the reference's). A periodic signal is instead
// continued by itself one period earlier, y[n+k] = y[n+k-P], with the
// period refined to a fraction of a sample (windowed-sinc delay). Used only
// when that continuation reproduces a held-out last quarter of the span to
// within -40 dB; otherwise returns false. Deterministic, O(n) after a
// decimated lag search.
inline bool periodicExtrapolate(const float* x, int n, int ext, float* out) {
    if (n < 512 || ext <= 0) return false;
    double e0 = 0.0;
    for (int i = 0; i < n; ++i) e0 += static_cast<double>(x[i]) * x[i];
    if (e0 < 1e-20) return false;
    const int H = n / 4;             // hold-out
    const int nt = n - H;            // training span for the hold-out test
    const int Pmin = 32, Pmax = nt / 2;
    if (Pmax <= Pmin + 4) return false;
    // normalised autocorrelation of the training span at lag P
    auto corr = [&](int P) {
        double s = 0.0, a = 0.0, b = 0.0;
        for (int i = P; i < nt; ++i) {
            s += static_cast<double>(x[i]) * x[i - P];
            a += static_cast<double>(x[i]) * x[i];
            b += static_cast<double>(x[i - P]) * x[i - P];
        }
        return s / (std::sqrt(a * b) + 1e-300);
    };
    // coarse search on a 4x decimated copy, then refine at full rate
    const int D = 4;
    std::vector<double> d(static_cast<size_t>(nt / D));
    for (size_t k = 0; k < d.size(); ++k) {
        double a = 0.0;
        for (int q = 0; q < D; ++q) a += x[k * D + q];
        d[k] = a / D;
    }
    int bestPd = -1;
    double bestRd = -2.0;
    const int nd = static_cast<int>(d.size());
    for (int Pd = Pmin / D; Pd <= Pmax / D; ++Pd) {
        double s = 0.0, a = 0.0, b = 0.0;
        for (int i = Pd; i < nd; ++i) {
            s += d[i] * d[i - Pd];
            a += d[i] * d[i];
            b += d[i - Pd] * d[i - Pd];
        }
        const double r = s / (std::sqrt(a * b) + 1e-300);
        if (r > bestRd) { bestRd = r; bestPd = Pd; }
    }
    if (bestPd < 0 || bestRd < 0.99) return false;
    int P = bestPd * D;
    double rP = -2.0;
    for (int c = std::max(Pmin, bestPd * D - D); c <= std::min(Pmax, bestPd * D + D); ++c) {
        const double r = corr(c);
        if (r > rP) { rP = r; P = c; }
    }
    if (rP < 0.99 || P <= Pmin || P >= Pmax) return false;
    // fractional refinement (parabola through the correlation peak)
    const double rm = corr(P - 1), rp = corr(P + 1);
    const double den = rm - 2.0 * rP + rp;
    double Pf = static_cast<double>(P);
    if (den < -1e-12) Pf += std::clamp(0.5 * (rm - rp) / den, -0.5, 0.5);
    // hold-out: continue the training span over the last quarter
    std::vector<double> y;
    detail::runPeriodic(x, nt, Pf, H, y);
    double err = 0.0, ref = 0.0;
    for (int k = 0; k < H; ++k) {
        const double dv = y[static_cast<size_t>(nt + k)] - x[nt + k];
        err += dv * dv;
        ref += static_cast<double>(x[nt + k]) * x[nt + k];
    }
    if (!(ref > 0.0) || err > 1e-4 * ref) return false;
    detail::runPeriodic(x, n, Pf, ext, y);
    for (int k = 0; k < ext; ++k) out[k] = static_cast<float>(y[static_cast<size_t>(n + k)]);
    return true;
}

// Edge extrapolation. With `periodic`, a strongly periodic edge is continued
// periodically (periodicExtrapolate; its lag search costs ~0.1 ms per
// channel, so the streaming-time head extension does not use it and only
// the end-of-stream paths do). Otherwise the covariance predictor is used
// when it extrapolates sanely (bounded growth), else the always-stable Burg
// predictor. Prediction
// reliability is measured by a hold-out test (train on all but the last n/4
// samples, extrapolate them, compare): a steady tone predicts to ~-100 dB, a
// chirp or noise does not. The extension is trusted for a horizon that
// shrinks with the hold-out error and then fades out smoothly, so an
// unpredictable edge degrades to (at worst) the old zero pad, never to
// confident garbage. Deterministic. Returns false when the edge had content
// but no trusted horizon at all (a glide, noise): what `out` then holds (a
// short faded burst of the prediction, or zeros with zeroUntrusted) is not a
// continuation of the signal.
inline bool burgExtrapolate(const float* x, int n, int order, int ext,
                            float* out, bool periodic = true,
                            bool zeroUntrusted = false) {
    for (int k = 0; k < ext; ++k) out[k] = 0.0f;
    if (periodic && periodicExtrapolate(x, n, ext, out)) return true;
    const int p = std::min(order, n / 8);
    if (p < 2 || n < 32) return true;
    double peak = 0.0, e0 = 0.0;
    for (int i = 0; i < n; ++i) {
        peak = std::max(peak, static_cast<double>(std::abs(x[i])));
        e0 += static_cast<double>(x[i]) * x[i];
    }
    if (e0 < 1e-20) return true;

    auto predict = [&](const float* tr, int nt, int count,
                       std::vector<double>& res) {
        std::vector<double> a;
        if (covariancePredictor(tr, nt, p, a) &&
            detail::runPredictor(tr, nt, a, p, count, 1.5 * peak, res)) {
            // late energy must not run away relative to the training energy
            double et = 0.0, ee = 0.0;
            for (int i = 0; i < nt; ++i) et += static_cast<double>(tr[i]) * tr[i];
            const int lateN = std::min(count, nt / 2);
            for (int k = count - lateN; k < count; ++k)
                ee += res[p + k] * res[p + k];
            if (lateN > 0 && ee / lateN <= 1.5 * et / nt) return true;
        }
        // fall back to the stable (but frequency-biased) Burg predictor
        std::vector<float> tmp(static_cast<size_t>(count));
        burgExtrapolateRaw(tr, nt, order, count, tmp.data());
        res.assign(static_cast<size_t>(p) + static_cast<size_t>(count), 0.0);
        for (int k = 0; k < count; ++k) res[p + k] = tmp[k];
        return true;
    };

    // hold-out reliability
    const int H = n / 4;
    std::vector<double> hres;
    predict(x, n - H, H, hres);
    double err = 0.0, ref = 0.0;
    for (int k = 0; k < H; ++k) {
        const double d = hres[p + k] - x[n - H + k];
        err += d * d;
        ref += static_cast<double>(x[n - H + k]) * x[n - H + k];
    }
    const double q = ref > 0.0 ? err / ref : 1.0;   // normalised hold-out error

    std::vector<double> res;
    predict(x, n, ext, res);

    // trusted horizon: full for q <= 1e-2, none for q >= 0.5 (log scale)
    double trust = 1.0;
    if (q > 1e-2)
        trust = std::max(0.0, 1.0 - std::log10(q / 1e-2) / std::log10(50.0));
    // No trusted horizon at all (a glide, noise): the predictor's output is
    // not a continuation of the signal. A caller whose consumer would take
    // even a faded burst of it for content (the analysis frames at the end of
    // a stream: measured on a log sweep's end, +3 dB overshoots crossing
    // 0 dBFS) asks for the plain zero pad instead (zeroUntrusted).
    if (zeroUntrusted && trust <= 0.0) return false;
    const double K0 = trust * ext;
    const double fade = 256.0;
    const float lim = static_cast<float>(1.05 * peak);
    for (int k = 0; k < ext; ++k) {
        double g = 1.0;
        if (k >= K0) {
            const double t = (k - K0) / fade;
            g = t >= 1.0 ? 0.0 : 0.5 * (1.0 + std::cos(3.14159265358979 * t));
        }
        float v = static_cast<float>(res[p + k] * g);
        out[k] = std::max(-lim, std::min(lim, v));
    }
    return trust > 0.0;
}

}  // namespace pbshift
