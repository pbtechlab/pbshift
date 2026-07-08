#!/usr/bin/env python3
"""Waveform-preserving pitch-synchronous voice time-stretch — WITS / PSMS arms.

Both arms share the TD-PSOLA front-end (robust_f0 / analysis_marks / smoothed
period contour from psola_stretch.py) but replace PSOLA's INTEGER-PERIOD COPY
with a genuinely interpolated pitch cycle, so intermediate periods that never
existed in the input are synthesized (no comb from repetition), while the REAL
mixed-phase pulse is carried (not regenerated from a lossy descriptor).

  Arm C  PSMS : time-domain normalized-period surface. Each pitch cycle is
                length-normalized to M samples; output cycles linearly
                interpolate the two neighbouring normalized cycles, then
                un-normalize to the local period. (Waveform Interpolation in the
                time domain.)
  Arm B  WITS : harmonic surface with a SEW/REW split. The normalized cycle's
                DFT is split into a Slowly-Evolving Waveform (short moving
                average across cycles = coherent voiced structure, interpolated)
                and a Rapidly-Evolving Waveform (residual = noise, its phase
                re-randomized per output cycle with a seeded PRNG so it never
                repeats). SEW+REW -> IDFT -> un-normalize.

Unvoiced spans in BOTH arms: REW-style regeneration — a grain with the local
magnitude spectrum but seeded-random phase, so noise is decorrelated (no flange).

Deterministic: all randomness from a counter-seeded PRNG.
usage: python benchmarks/wits_psms_stretch.py in.wav out.wav ALPHA [wits|psms]
"""
import sys
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import psola_stretch as ps

M = 512  # normalized cycle length (represents ~2 pitch periods)


def resample_to(v, L):
    """BANDLIMITED resample (polyphase FIR) v -> length L. Linear interpolation
    aliases high frequencies into broadband noise on the normalize/un-normalize
    round-trip; a bandlimited resampler is essential for clean voice."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) == L or len(v) < 2:
        return v if len(v) == L else np.interp(
            np.linspace(0, 1, L), np.linspace(0, 1, len(v)), v)
    g = math.gcd(L, len(v))
    up, down = L // g, len(v) // g
    return resample_poly(v, up, down)


def extract_cws(x, marks, per, pol):
    """Normalized 2-period characteristic waveforms, each circularly rolled so
    its consistent-polarity main pulse sits at M/2. Peak-centering ALIGNS the
    cycles so interpolating neighbours blends one pulse onto one pulse (no
    chorus/doubling) instead of two misaligned pulses."""
    n = len(x)
    cws = np.zeros((len(marks), M))
    for i, mk in enumerate(marks):
        T = int(round(per[min(mk, n - 1)]))
        a, b = mk - T, mk + T
        seg = np.zeros(2 * T)
        ga, gb = max(0, a), min(n, b)
        seg[(ga - a):(ga - a) + (gb - ga)] = x[ga:gb]
        cw = resample_to(seg, M)
        pk = int(np.argmax(cw * pol))
        cws[i] = np.roll(cw, M // 2 - pk)
    return cws


def prng_phase(seed, k):
    """Deterministic per-(cycle,bin) random phase in [-pi, pi) via splitmix64."""
    z = (np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) +
         np.arange(k, dtype=np.uint64) + np.uint64(0x632BE59BD9B4E019))
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z.astype(np.float64) / 2.0**64) * 2.0 * np.pi - np.pi


def ola(out, norm, grain, t_out):
    L = len(grain)
    w = np.hanning(L)
    oa = int(round(t_out)) - L // 2
    oi = oa + np.arange(L)
    m = (oi >= 0) & (oi < len(out))
    np.add.at(out, oi[m], grain[m] * w[m])
    np.add.at(norm, oi[m], w[m])


def rew_noise(x, center, T, seed):
    """Unvoiced grain: local magnitude spectrum, seeded-random phase (decorrelated)."""
    n = len(x)
    a, b = int(center) - T, int(center) + T
    seg = np.zeros(2 * T)
    ga, gb = max(0, a), min(n, b)
    seg[(ga - a):(ga - a) + (gb - ga)] = x[ga:gb]
    S = np.fft.rfft(seg * np.hanning(len(seg)))
    ph = prng_phase(seed, len(S))
    ph[0] = 0.0
    return np.fft.irfft(np.abs(S) * np.exp(1j * ph), len(seg))


def stretch(x, sr, alpha, arm, fmin=70.0, fmax=500.0, hop=256):
    x = x.astype(np.float64)
    n = len(x)
    f0s, voiced = ps.robust_f0(x, sr, fmin, fmax, hop)
    per = ps.smooth_period(f0s, voiced, sr, fmin, fmax)
    pol = ps.excitation_polarity(x)
    marks = ps.analysis_marks(x, voiced, per, pol)
    if len(marks) < 3:
        return np.zeros(int(round(n * alpha)))

    cws = extract_cws(x, marks, per, pol)            # (K, M), peak-aligned
    Hsurf = np.fft.rfft(cws, axis=1)                 # (K, M//2+1)
    # SEW = moving average over +-win cycles; REW = residual (aperiodic detail)
    K = len(marks)
    win = 3
    sew = np.zeros_like(Hsurf)
    for i in range(K):
        lo, hi = max(0, i - win), min(K, i + win + 1)
        sew[i] = Hsurf[lo:hi].mean(axis=0)
    rew = Hsurf - sew

    out_len = int(round(n * alpha))
    pad = int(np.max(per)) + 8
    out = np.zeros(out_len + pad)
    norm = np.zeros_like(out)

    cyc = 0
    t_out = float(marks[0])
    while t_out < out_len:
        src = t_out / alpha
        si = int(np.clip(src, 0, n - 1))
        T = max(2, int(round(per[si])))
        if voiced[si]:
            # continuous analysis-cycle index + fraction
            i = int(np.clip(np.searchsorted(marks, src) - 1, 0, K - 2))
            span = max(1.0, float(marks[i + 1] - marks[i]))
            frac = float(np.clip((src - marks[i]) / span, 0.0, 1.0))
            if arm == "psms":
                # time-domain faithful interpolation of aligned cycles
                cw = (1.0 - frac) * cws[i] + frac * cws[i + 1]
            else:  # wits: frequency-domain interpolation. SEW carried faithfully;
                # REW (aperiodic detail) phase-regenerated ONLY in proportion to
                # local aperiodicity so clean voiced stays coherent (no buzz).
                sew_i = (1.0 - frac) * sew[i] + frac * sew[i + 1]
                rew_i = (1.0 - frac) * rew[i] + frac * rew[i + 1]
                aper = float(np.clip(np.linalg.norm(rew_i) /
                                     (np.linalg.norm(sew_i) + 1e-9), 0.0, 1.0))
                if aper > 0.35:  # breathy/noisy cycle -> decorrelate the residual
                    ph = prng_phase((cyc * 2654435761) & 0xFFFFFFFF, len(rew_i))
                    rew_i = np.abs(rew_i) * np.exp(1j * ph)
                cw = np.fft.irfft(sew_i + rew_i, M)
            grain = resample_to(cw, 2 * T)
            ola(out, norm, grain, t_out)
        else:
            grain = rew_noise(x, src, T, cyc * 40503 & 0xFFFFFFFF)
            ola(out, norm, grain, t_out)
        t_out += T
        cyc += 1

    return out[:out_len] / np.maximum(norm[:out_len], 1e-6)


def main():
    inp, outp, alpha = sys.argv[1], sys.argv[2], float(sys.argv[3])
    arm = sys.argv[4] if len(sys.argv) > 4 else "wits"
    x, sr = sf.read(inp)
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = stretch(x, sr, alpha, arm)
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 1.0:
        y = y / peak * 0.99
    sf.write(outp, y.astype(np.float32), sr)
    print(f"[{arm}] {inp} -> {outp}  alpha={alpha}  {len(x)}->{len(y)} "
          f"({len(y)/sr:.3f}s) peak={peak:.3f}")


if __name__ == "__main__":
    main()
