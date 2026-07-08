#!/usr/bin/env python3
"""HND time-stretch: harmonic-keep + decorrelated-residual TD-PSOLA (candidate #9).

The residual audible flange in clean TD-PSOLA is the DUPLICATED VOICED APERIODIC
RESIDUAL (breath/aspiration) combing — the harmonic-part duplication is inaudible
(its comb notches fall between harmonics) and the unvoiced flange is already
handled. So: split each voiced grain into a HARMONIC part (pitch-synchronous
average of neighbouring real periods — kept as verbatim real copies) and a
RESIDUAL part (aperiodic), and DECORRELATE only the residual with a magnitude-
preserving operation, so its duplicates stop combing WITHOUT adding a broadband
floor into the spectral valleys (the WI failure mode).

Reuses the psola_stretch front-end; works at NATIVE period (no resample round-
trip — that was the WI aliasing-floor source). Deterministic.

decorr modes:
  jit : residual source jitter (draw duplicates from different REAL positions;
        real content -> valley-clean, no overlap-cancellation)
  ap  : residual per-grain magnitude-preserving phase scramble (all-pass)

usage: python benchmarks/hnd_stretch.py in.wav out.wav ALPHA [jit|ap]
"""
import sys
import numpy as np
import soundfile as sf

import psola_stretch as ps


def prng(seed):
    z = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(0x632BE59BD9B4E019)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return float(z) / 2.0**64


def harmonic_residual(x, marks, per, voiced, N=3):
    """Pitch-synchronous split: h = average of aligned neighbour periods (the
    coherent harmonic part), r = x - h (the aperiodic residual)."""
    n = len(x)
    h = np.zeros(n)
    nrm = np.zeros(n)
    for idx, mk in enumerate(marks):
        if not voiced[min(mk, n - 1)]:
            continue
        T = max(2, int(round(per[min(mk, n - 1)])))
        acc = np.zeros(2 * T)
        cnt = 0
        for j in range(max(0, idx - N), min(len(marks), idx + N + 1)):
            mj = marks[j]
            a, b = mj - T, mj + T
            g = np.zeros(2 * T)
            ga, gb = max(0, a), min(n, b)
            g[(ga - a):(ga - a) + (gb - ga)] = x[ga:gb]
            acc += g
            cnt += 1
        hg = acc / max(1, cnt)
        w = np.hanning(2 * T)
        oi = np.arange(2 * T) + (mk - T)
        m = (oi >= 0) & (oi < n)
        np.add.at(h, oi[m], hg[m] * w[m])
        np.add.at(nrm, oi[m], w[m])
    h = h / np.maximum(nrm, 1e-6)
    return h, x - h


def add_grain(out, norm, center, L, sig, t_out, randomize=False, seed=0):
    n = len(sig)
    a, b = int(center) - L, int(center) + L
    ga, gb = max(0, a), min(n, b)
    if gb - ga < 4:
        return
    seg = sig[ga:gb].astype(np.float64).copy()
    if randomize and len(seg) >= 8:
        S = np.fft.rfft(seg)
        ph = np.array([prng((seed * 2654435761 + k) & 0xFFFFFFFFFFFF)
                       for k in range(len(S))]) * 2.0 * np.pi
        ph[0] = 0.0
        if len(seg) % 2 == 0:
            ph[-1] = 0.0
        seg = np.fft.irfft(np.abs(S) * np.exp(1j * ph), len(seg))
    w = np.hanning(len(seg))
    oa = int(round(t_out)) - (int(center) - ga)
    oi = oa + np.arange(len(seg))
    m = (oi >= 0) & (oi < len(out))
    np.add.at(out, oi[m], seg[m] * w[m])
    np.add.at(norm, oi[m], w[m])


def stretch_stream(sig, marks, per, voiced, alpha, decorr=None):
    n = len(sig)
    out_len = int(round(n * alpha))
    out = np.zeros(out_len + int(np.max(per)) + 8)
    norm = np.zeros_like(out)
    t_out = float(marks[0])
    cyc = 0
    while t_out < out_len:
        src = t_out / alpha
        si = int(np.clip(src, 0, n - 1))
        T = float(per[si])
        L = max(2, int(round(T)))
        tol = max(4, int(0.4 * T))
        if voiced[si]:
            i = int(np.argmin(np.abs(marks - src)))
            center = marks[i]
        else:
            center = int(round(src))
        randomize = False
        if decorr == "jit" and voiced[si]:
            center = int(round(center + (prng((cyc * 40503) & 0xFFFFFFFF) - 0.5) * T))
        elif decorr == "ap" and voiced[si]:
            randomize = True
        d = 0 if decorr else ps.best_offset(out, norm, center, L, sig, t_out, tol)
        add_grain(out, norm, center + d, L, sig, t_out, randomize=randomize, seed=cyc)
        t_out += T
        cyc += 1
    return out[:out_len] / np.maximum(norm[:out_len], 1e-6)


def hnd_timestretch(x, sr, alpha, decorr="jit", fmin=70.0, fmax=500.0, hop=256):
    x = x.astype(np.float64)
    f0s, voiced = ps.robust_f0(x, sr, fmin, fmax, hop)
    per = ps.smooth_period(f0s, voiced, sr, fmin, fmax)
    pol = ps.excitation_polarity(x)
    marks = ps.analysis_marks(x, voiced, per, pol)
    if len(marks) < 3:
        return np.zeros(int(round(len(x) * alpha)))
    h, r = harmonic_residual(x, marks, per, voiced)
    H = stretch_stream(h, marks, per, voiced, alpha, decorr=None)     # harmonic: real copies
    R = stretch_stream(r, marks, per, voiced, alpha, decorr=decorr)   # residual: decorrelated
    m = min(len(H), len(R))
    return H[:m] + R[:m]


def main():
    inp, outp, alpha = sys.argv[1], sys.argv[2], float(sys.argv[3])
    decorr = sys.argv[4] if len(sys.argv) > 4 else "jit"
    x, sr = sf.read(inp)
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = hnd_timestretch(x, sr, alpha, decorr=decorr)
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 1.0:
        y = y / peak * 0.99
    sf.write(outp, y.astype(np.float32), sr)
    print(f"[hnd:{decorr}] {inp} -> {outp}  {len(x)}->{len(y)} ({len(y)/sr:.3f}s) peak={peak:.3f}")


if __name__ == "__main__":
    main()
