#!/usr/bin/env python3
"""Clean-room TD-PSOLA + WSOLA-lite time-stretch for voice.

Pure time-domain (no FFT phase) => structurally free of phase-vocoder
"phasiness / fine DelayEcho". Pitch preserved by keeping the local pitch period;
length changed by repeating/dropping whole periods in VOICED spans.

Anti-artifact measures (each targets a specific audible defect):
  * UNVOICED flange (periodic noise repeat): advance the source pointer
    CONTINUOUSLY instead of repeating grains.
  * clicks from phase cancellation / grain-boundary jumps: WSOLA-style
    correlation alignment places every grain to add CONSTRUCTIVELY.
  * clicks from pitch-mark POLARITY flips: marks are picked with a single,
    globally-consistent excitation polarity.
  * clicks at voiced/unvoiced boundaries: the pitch-period contour is smoothed
    and carried through unvoiced spans so grain sizes stay continuous.

F0 via librosa.pyin (pYIN), median-smoothed with an octave-jump guard.
usage: python benchmarks/psola_stretch.py in.wav out.wav ALPHA
"""
import sys
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import medfilt


def robust_f0(x, sr, fmin, fmax, hop):
    f0, vflag, _ = librosa.pyin(x, fmin=fmin, fmax=fmax, sr=sr,
                                frame_length=2048, hop_length=hop)
    voiced = np.nan_to_num(vflag, nan=0.0).astype(bool)
    f = f0.copy()
    fv = np.where(voiced & np.isfinite(f), f, np.nan)
    if np.isfinite(fv).sum() > 5:
        idx = np.where(np.isfinite(fv))[0]
        sm = medfilt(fv[idx], kernel_size=min(7, len(idx) | 1))
        for jj, j in enumerate(idx):
            if sm[jj] > 0:
                r = f[j] / sm[jj]
                if r > 1.6:
                    f[j] /= 2.0
                elif r < 0.6:
                    f[j] *= 2.0
    n = len(x)
    fidx = np.clip(np.arange(n) // hop, 0, len(f) - 1)
    return f[fidx], voiced[fidx]


def smooth_period(f0s, voiced, sr, fmin, fmax, default_hz=150.0):
    """Per-sample pitch period (samples), filled/held through unvoiced spans so
    grain sizes stay continuous across voiced/unvoiced boundaries."""
    per = np.where(voiced & np.isfinite(f0s) & (f0s > 0),
                   sr / np.maximum(f0s, 1e-6), np.nan)
    n = len(per)
    if np.isfinite(per).any():
        idx = np.arange(n)
        good = np.isfinite(per)
        per = np.interp(idx, idx[good], per[good])  # hold/interp through gaps
    else:
        per[:] = sr / default_hz
    return np.clip(per, sr / fmax, sr / fmin)


def excitation_polarity(x):
    """Glottal excitation tends to have a consistent-sign sharp peak. Use the
    signal's cubed-mean sign as a cheap global polarity estimate."""
    return 1.0 if np.mean(x ** 3) >= 0.0 else -1.0


def analysis_marks(x, voiced, per, pol):
    """One analysis mark per pitch period, refined to the local extremum of the
    CONSISTENT polarity (pol) — avoids polarity-flip clicks between grains."""
    n = len(x)
    marks = []
    pos = int(round(per[0]))
    while pos < n - 1:
        T = float(per[min(pos, n - 1)])
        if voiced[min(pos, n - 1)]:
            w = max(2, int(0.3 * T))
            a, b = max(0, pos - w), min(n, pos + w)
            seg = x[a:b] * pol
            pos_ref = a + int(np.argmax(seg))
        else:
            pos_ref = pos
        marks.append(pos_ref)
        pos = pos_ref + int(round(T))
    return np.array(marks, dtype=int)


def add_grain(out, norm, center_src, L, x, t_out, reverse=False):
    n = len(x)
    a, b = int(center_src) - L, int(center_src) + L
    ga, gb = max(0, a), min(n, b)
    if gb - ga < 4:
        return
    seg = x[ga:gb].copy()
    if reverse:
        # time-reversal decorrelates a duplicated noise grain from its neighbour
        # (breaks the unvoiced comb) while preserving its magnitude spectrum
        # exactly (valley-clean: adds no broadband floor).
        seg = seg[::-1]
    w = np.hanning(len(seg))
    oa = int(round(t_out)) - (int(center_src) - ga)
    oi = oa + np.arange(len(seg))
    m = (oi >= 0) & (oi < len(out))
    np.add.at(out, oi[m], seg[m] * w[m])
    np.add.at(norm, oi[m], w[m])


def best_offset(out, norm, center_src, L, x, t_out, tol, step=2):
    """WSOLA alignment: shift the source grain by delta in [-tol,tol] so its
    leading half adds CONSTRUCTIVELY to the output already at t_out."""
    ta, tb = int(round(t_out)) - L, int(round(t_out))
    ta_c = max(0, ta)
    if tb - ta_c < 8:
        return 0
    nn = norm[ta_c:tb]
    tmpl = np.where(nn > 1e-6, out[ta_c:tb] / np.maximum(nn, 1e-6), 0.0)
    if np.linalg.norm(tmpl) < 1e-9:
        return 0
    best_d, best_c = 0, -1e18
    for d in range(-tol, tol + 1, step):
        cs = int(center_src) + d
        a_c = max(0, cs - L)
        seg = x[a_c:cs]
        m = min(len(seg), len(tmpl))
        if m < 8:
            continue
        s = seg[-m:]
        c = float(np.dot(s, tmpl[-m:])) / (np.linalg.norm(s) + 1e-9)
        if c > best_c:
            best_c, best_d = c, d
    return best_d


def psola_timestretch(x, sr, alpha, fmin=70.0, fmax=500.0, hop=256):
    x = x.astype(np.float64)
    n = len(x)
    f0s, voiced = robust_f0(x, sr, fmin, fmax, hop)
    per = smooth_period(f0s, voiced, sr, fmin, fmax)
    pol = excitation_polarity(x)
    marks = analysis_marks(x, voiced, per, pol)
    if len(marks) < 2:
        return np.zeros(int(round(n * alpha)))

    out_len = int(round(n * alpha))
    pad = int(np.max(per)) + 8
    out = np.zeros(out_len + pad)
    norm = np.zeros_like(out)

    t_out = float(marks[0])
    uv_cyc = 0
    while t_out < out_len:
        src = t_out / alpha
        si = int(np.clip(src, 0, n - 1))
        T = float(per[si])
        L = max(2, int(round(T)))
        tol = max(4, int(0.4 * T))
        if voiced[si]:
            # WIDE WSOLA search on voiced: at 2x each pitch period is reused, and
            # a NARROW search locks the reuse onto a near-identical copy = the
            # deepest duplicate-delay comb (the audible "doubling/flange"). A wide
            # (+-0.9*T) search lets the reuse land on the best-matching REAL
            # position instead, breaking the exact-copy comb while staying real
            # (no chorus) and coherent (no click). Measured: comb 0.0127 -> ~0.009
            # AND valley stays clean (<=1.34) -> the only lever that cut the
            # doubling without adding chorus (morph) or noise (decorrelation).
            i = int(np.argmin(np.abs(marks - src)))
            center = marks[i]
            d = best_offset(out, norm, center, L, x, t_out, max(4, int(0.9 * T)))
            add_grain(out, norm, center + d, L, x, t_out)
            uv_cyc = 0
        else:
            # UNVOICED: continuous source (never exactly repeat) + correlation
            # alignment. NOTE: time-reversal decorrelation was tried here and
            # REVERTED — it stayed valley-clean but glitched (momentary noise) on
            # voicing-misdetected / isolated unvoiced grains, a net regression
            # vs the champion PSOLA v3.
            center = int(round(src))
            d = best_offset(out, norm, center, L, x, t_out, tol)
            add_grain(out, norm, center + d, L, x, t_out)
            uv_cyc += 1
        t_out += T

    y = out[:out_len] / np.maximum(norm[:out_len], 1e-6)
    return y


def main():
    inp, outp, alpha = sys.argv[1], sys.argv[2], float(sys.argv[3])
    x, sr = sf.read(inp)
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = psola_timestretch(x, sr, alpha)
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 1.0:
        y = y / peak * 0.99
    sf.write(outp, y.astype(np.float32), sr)
    print(f"{inp} -> {outp}  alpha={alpha}  {len(x)} -> {len(y)} samples "
          f"({len(y)/sr:.3f}s)  peak={peak:.3f}")


if __name__ == "__main__":
    main()
