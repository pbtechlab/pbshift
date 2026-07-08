#!/usr/bin/env python3
"""PS-HNM/R time-stretch: continuous-phase harmonic bank + REAL residual.

The last structurally-different family. The voiced signal below a Maximum Voiced
Frequency (MVF) is resynthesised as ONE phase-integrated oscillator bank at
k*f0(t): length is made by INTEGRATING phase on the output clock, never by
duplicating or blending a period -> the duplicate-delay comb (PSOLA) and the
neighbour-blend chorus (morph) cannot exist by construction. The aperiodic part
(above MVF + the unvoiced spans) is carried as the REAL residual, WSOLA-stretched
-> valley-clean (no synthesised broadband floor), and it re-adds the breath that
a pure harmonic bank lacks (the "de-breathed buzz" cure).

Shape-invariant phase (SHIP): each harmonic's synthesis phase is k*phi1 + the
MEASURED relative phase (theta_k - k*theta_1), so the glottal pulse SHAPE is
preserved instead of collapsing to an all-cosine buzz.

Deterministic. usage: python benchmarks/hnm_stretch.py in.wav out.wav ALPHA
"""
import sys
import numpy as np
import soundfile as sf

import psola_stretch as ps

HA = 200          # analysis hop (samples)
MVF_HZ = 5000.0   # max voiced frequency (harmonics above -> residual band)


def unwrap_cols(P):
    return np.unwrap(P, axis=0)


def analyze(x, sr, f0s, voiced):
    """Per analysis frame: LEAST-SQUARES harmonic fit at k*f0 (k=1..K<=MVF).

    x[n] ~= sum_k [a_k cos(k w0 tau) + b_k sin(k w0 tau)] solved by weighted LS
    over ~2.5 periods. LS is accurate even when the window isn't an integer
    number of periods (unlike DFT-at-k*f0, whose inter-harmonic leakage left
    harmonic energy in the residual -> valley fill / the +2.98 dB failure).
    Amplitude A_k=|c_k|, phase theta_k with harmonic = A_k cos(k w0 tau + th_k)."""
    n = len(x)
    centers = np.arange(0, n, HA)
    f0pos = f0s[f0s > 0]
    Kmax = int(MVF_HZ / (np.nanmin(f0pos) if f0pos.size else 100.0)) + 2
    Kmax = min(Kmax, 200)
    A = np.zeros((len(centers), Kmax))
    TH = np.zeros((len(centers), Kmax))
    F0 = np.zeros(len(centers))
    VOI = np.zeros(len(centers), bool)
    for fi, c in enumerate(centers):
        ci = min(c, n - 1)
        f0 = f0s[ci] if (voiced[ci] and f0s[ci] > 0) else 0.0
        F0[fi] = f0
        VOI[fi] = f0 > 0
        if f0 <= 0:
            continue
        T = sr / f0
        W = int(round(2.5 * T))
        idx = np.arange(c - W, c + W)
        seg = np.where((idx >= 0) & (idx < n), x[np.clip(idx, 0, n - 1)], 0.0)
        win = np.hanning(len(seg))
        tau = (idx - c) / sr
        K = min(Kmax, int(MVF_HZ / f0))
        if K < 1:
            continue
        ph = 2 * np.pi * f0 * np.outer(tau, np.arange(1, K + 1))   # (len, K)
        B = np.empty((len(idx), 2 * K))
        B[:, 0::2] = np.cos(ph)
        B[:, 1::2] = np.sin(ph)
        Bw = B * win[:, None]
        coef, *_ = np.linalg.lstsq(Bw, seg * win, rcond=None)
        a_k, b_k = coef[0::2], coef[1::2]
        A[fi, :K] = np.hypot(a_k, b_k)
        TH[fi, :K] = np.arctan2(-b_k, a_k)          # x = A cos(k w0 tau + th)
    return centers, F0, VOI, A, TH


def synth_harmonic(centers, F0, VOI, A, TH, n_out, sr, alpha):
    """Oscillator-bank synthesis on the output clock with SHIP relative phases."""
    Kmax = A.shape[1]
    # map each output sample to an analysis time, then to a fractional frame index
    t_an = np.arange(n_out) / alpha
    fpos = t_an / HA
    fi0 = np.clip(np.floor(fpos).astype(int), 0, len(centers) - 2)
    fr = fpos - fi0
    # instantaneous f0 per output sample (interp), phase integrated on output clock
    f0o = F0[fi0] * (1 - fr) + F0[fi0 + 1] * fr
    voi_o = VOI[fi0] & VOI[fi0 + 1]
    f0o = np.where(voi_o, f0o, 0.0)
    phi1 = np.cumsum(2 * np.pi * f0o / sr)
    # relative phase (theta_k - k*theta_1), unwrapped across frames, interpolated
    th1 = TH[:, 0]
    rel = TH - (np.arange(1, Kmax + 1)[None, :] * th1[:, None])
    rel = np.angle(np.exp(1j * rel))                # principal value
    out = np.zeros(n_out)
    for k in range(1, Kmax + 1):
        Ak = A[fi0, k - 1] * (1 - fr) + A[fi0 + 1, k - 1] * fr
        if Ak.max() < 1e-6:
            continue
        rk = rel[fi0, k - 1] * (1 - fr) + rel[fi0 + 1, k - 1] * fr
        out += np.where(voi_o, Ak * np.cos(k * phi1 + rk), 0.0)
    return out


def synth_harmonic_analysis(centers, F0, VOI, A, TH, n, sr):
    """Same bank at ANALYSIS timing (alpha=1) to form the residual x - h."""
    return synth_harmonic(centers, F0, VOI, A, TH, n, sr, 1.0)


def wsola_residual(r, marks, per, voiced, alpha):
    """Stretch the real residual by continuous-source PSOLA (valley-clean)."""
    n = len(r)
    out_len = int(round(n * alpha))
    out = np.zeros(out_len + int(np.max(per)) + 8)
    norm = np.zeros_like(out)
    t_out = float(marks[0]) if len(marks) else 0.0
    while t_out < out_len:
        src = t_out / alpha
        si = int(np.clip(src, 0, n - 1))
        T = float(per[si])
        L = max(2, int(round(T)))
        tol = max(4, int(0.4 * T))
        center = int(round(src))
        d = ps.best_offset(out, norm, center, L, r, t_out, tol)
        ps.add_grain(out, norm, center + d, L, r, t_out)
        t_out += T
    return out[:out_len] / np.maximum(norm[:out_len], 1e-6)


def hnm_timestretch(x, sr, alpha, fmin=70.0, fmax=500.0, hop=256):
    x = x.astype(np.float64)
    n = len(x)
    f0s, voiced = ps.robust_f0(x, sr, fmin, fmax, hop)
    per = ps.smooth_period(f0s, voiced, sr, fmin, fmax)
    pol = ps.excitation_polarity(x)
    marks = ps.analysis_marks(x, voiced, per, pol)
    centers, F0, VOI, A, TH = analyze(x, sr, f0s, voiced)
    h_an = synth_harmonic_analysis(centers, F0, VOI, A, TH, n, sr)
    r = x - h_an                                    # real residual (breath + HF + errors)
    out_len = int(round(n * alpha))
    H = synth_harmonic(centers, F0, VOI, A, TH, out_len, sr, alpha)
    R = wsola_residual(r, marks, per, voiced, alpha)
    m = min(len(H), len(R))
    return H[:m] + R[:m]


def main():
    inp, outp, alpha = sys.argv[1], sys.argv[2], float(sys.argv[3])
    x, sr = sf.read(inp)
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = hnm_timestretch(x, sr, alpha)
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 1.0:
        y = y / peak * 0.99
    sf.write(outp, y.astype(np.float32), sr)
    print(f"[hnm] {inp} -> {outp}  {len(x)}->{len(y)} ({len(y)/sr:.3f}s) peak={peak:.3f}")


if __name__ == "__main__":
    main()
