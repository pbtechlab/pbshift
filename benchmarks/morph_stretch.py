#!/usr/bin/env python3
"""morph_stretch.py -- PIVOT-lite: whole-grain convex period MORPH (no split).

Panel probe for the PIVOT / Fractional-epoch-MORPHING family. Instead of
duplicating the nearest real period (PSOLA's fixed-lag identical copy = comb),
synthesize each voiced output period as a convex blend of the TWO bracketing
REAL periods:  g_out = (1-f)*g_i + f*g_{i+1}, epoch-aligned.

Valley-clean by convex hull: |G_out| <= (1-f)|G_i| + f|G_{i+1}| at every bin,
so where both real neighbours are null (input valley) the blend is null. NO
harmonic/residual split, NO phase scramble -> none of the split-leakage
valley-fill that sank hnd_stretch (jit +5.9 dB, ap +9.5 dB).

usage: python morph_stretch.py in.wav out.wav ALPHA
"""
import sys
import numpy as np
import soundfile as sf
import psola_stretch as ps


def add_seg(out, norm, seg, w, center_off, t_out):
    oa = int(round(t_out)) - center_off
    oi = oa + np.arange(len(seg))
    m = (oi >= 0) & (oi < len(out))
    np.add.at(out, oi[m], seg[m] * w[m])
    np.add.at(norm, oi[m], w[m])


def grain(x, center, L):
    n = len(x)
    a, b = int(center) - L, int(center) + L
    ga, gb = max(0, a), min(n, b)
    g = np.zeros(2 * L)
    g[(ga - a):(ga - a) + (gb - ga)] = x[ga:gb]
    return g, L  # center offset = L (mark-centered)


def grain_frac(x, center_f, L):
    """Mark-centered grain with SUB-SAMPLE center (linear fractional delay)."""
    idx = np.arange(-L, L, dtype=np.float64) + center_f
    return np.interp(idx, np.arange(len(x), dtype=np.float64), x, left=0.0, right=0.0)


# STEADY_THRESH: above this normalized neighbour correlation the two periods are
# ~identical (steady vowel) -> blending buys no de-comb but costs valley from
# residual cancellation, so fall back to a single REAL grain. Only morph where
# periods genuinely drift (which is exactly where duplication would comb).
STEADY_THRESH = 0.97


def morph_timestretch(x, sr, alpha, fmin=70.0, fmax=500.0, hop=256):
    x = x.astype(np.float64)
    n = len(x)
    f0s, voiced = ps.robust_f0(x, sr, fmin, fmax, hop)
    per = ps.smooth_period(f0s, voiced, sr, fmin, fmax)
    pol = ps.excitation_polarity(x)
    marks = ps.analysis_marks(x, voiced, per, pol)
    if len(marks) < 3:
        return np.zeros(int(round(n * alpha)))
    out_len = int(round(n * alpha))
    pad = int(np.max(per)) + 8
    out = np.zeros(out_len + pad)
    norm = np.zeros_like(out)
    t_out = float(marks[0])
    while t_out < out_len:
        src = t_out / alpha
        si = int(np.clip(src, 0, n - 1))
        T = float(per[si])
        L = max(2, int(round(T)))
        tol = max(4, int(0.4 * T))
        if voiced[si]:
            # bracketing marks around src
            i = int(np.searchsorted(marks, src) - 1)
            i = max(0, min(i, len(marks) - 2))
            m0, m1 = marks[i], marks[i + 1]
            f = 0.0 if m1 <= m0 else np.clip((src - m0) / (m1 - m0), 0.0, 1.0)
            both_v = voiced[min(m0, n - 1)] and voiced[min(m1, n - 1)]
            if both_v:
                g0, _ = grain(x, m0, L)
                # epoch-align g1 to g0: integer cross-correlation search, then
                # parabolic sub-sample refine + fractional-delay extraction, so
                # excitation epochs + low harmonics coincide -> minimal blend
                # cancellation (the source of the +0.29 dB valley residue).
                cc = np.array([float(np.dot(g0, grain(x, m1 + dd, L)[0]))
                               for dd in range(-tol, tol + 1)])
                k = int(np.argmax(cc))
                bd = k - tol
                if 0 < k < len(cc) - 1:
                    a0, b0, c0v = cc[k - 1], cc[k], cc[k + 1]
                    den = (a0 - 2 * b0 + c0v)
                    if abs(den) > 1e-18:
                        bd += float(np.clip(0.5 * (a0 - c0v) / den, -0.5, 0.5))
                g0n = np.linalg.norm(g0) + 1e-12
                g1 = grain_frac(x, m1 + bd, L)
                corr = float(np.dot(g0, g1)) / (g0n * (np.linalg.norm(g1) + 1e-12))
                if corr >= STEADY_THRESH:
                    # steady vowel: neighbours ~identical -> single real grain
                    seg = g0
                else:
                    seg = (1.0 - f) * g0 + f * g1
                add_seg(out, norm, seg, np.hanning(2 * L), L, t_out)
            else:
                d = ps.best_offset(out, norm, m0, L, x, t_out, tol)
                g0, c0 = grain(x, m0 + d, L)
                add_seg(out, norm, g0, np.hanning(2 * L), L, t_out)
        else:
            center = int(round(src))
            d = ps.best_offset(out, norm, center, L, x, t_out, tol)
            g0, c0 = grain(x, center + d, L)
            add_seg(out, norm, g0, np.hanning(2 * L), L, t_out)
        t_out += T
    return out[:out_len] / np.maximum(norm[:out_len], 1e-6)


def main():
    inp, outp, alpha = sys.argv[1], sys.argv[2], float(sys.argv[3])
    x, sr = sf.read(inp)
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = morph_timestretch(x, sr, alpha)
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 1.0:
        y = y / peak * 0.99
    sf.write(outp, y.astype(np.float32), sr)
    print(f"[morph] {inp} -> {outp}  {len(x)}->{len(y)} ({len(y)/sr:.3f}s) peak={peak:.3f}")


if __name__ == "__main__":
    main()
