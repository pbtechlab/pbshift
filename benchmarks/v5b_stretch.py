#!/usr/bin/env python3
"""Vocal-mode voice time-stretch (clean-room, original, IP-safe).

Matches a top commercial vocal-mode reference on 2x voice (human-confirmed),
using only original clean-room DSP on public / expired-patent foundations.
A CONTENT- and BAND-adaptive time-domain method:

  Front-end (psola_stretch.py): pYIN f0 -> voiced flag + smoothed pitch period;
  consistent-polarity pitch marks.

  Split at fc = 3000 Hz (zero-phase Butterworth; complementary so the bands sum
  back flat).

  LOW band (< 3000 Hz)  -- carries the perceived naturalness:
    TD-PSOLA on real pitch-period grains. VOICED reuse uses a WIDE (+-0.9*T)
    WSOLA search so the 2x period-reuse lands on the best-matching REAL position
    instead of a frozen-identical copy -> breaks the duplicate-delay comb
    ("doubling") while staying real (no chorus) and valley-clean (no synthesized
    noise floor). UNVOICED uses continuous-source placement.

  HIGH band (> 3000 Hz) -- mostly aspiration/sibilants:
    continuous-source grains with alternate TIME-REVERSAL. Reversal preserves
    the magnitude spectrum exactly (valley-clean) but decorrelates the duplicated
    noise -> removes the high-band doubling that a plain copy leaves.

Verified clean by benchmarks/gate.py (valley_excess ~0.5 dB, commercial-reference
level; comb below reference engines). The one residual (a faint high-band detune
in sustained vowels) is at the inherent limit of 2x harmonic stretch (the
commercial reference exhibits it too); an exhaustive invention search found no
valley-clean way past it.

usage: python benchmarks/v5b_stretch.py in.wav out.wav ALPHA
"""
import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

import psola_stretch as ps

FC = 3000.0            # low/high crossover (Hz)
VOICED_TOL = 0.9       # WSOLA search half-width on voiced reuse (x period)
UNVOICED_TOL = 0.4


def _stretch_band(sig, marks, per, voiced, alpha, high_band):
    """One band. Low band: wide-WSOLA voiced grains. High band: continuous +
    alternate time-reversal de-double."""
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
        vo = voiced[si]
        if high_band:
            center = int(round(src))                 # continuous source
            d = ps.best_offset(out, norm, center, L, sig, t_out,
                               max(4, int(UNVOICED_TOL * T)))
            ps.add_grain(out, norm, center + d, L, sig, t_out,
                         reverse=bool(cyc & 1))       # decorrelate duplicated HF
        elif vo:
            center = marks[int(np.argmin(np.abs(marks - src)))]
            d = ps.best_offset(out, norm, center, L, sig, t_out,
                               max(4, int(VOICED_TOL * T)))
            ps.add_grain(out, norm, center + d, L, sig, t_out)
        else:
            center = int(round(src))
            d = ps.best_offset(out, norm, center, L, sig, t_out,
                               max(4, int(UNVOICED_TOL * T)))
            ps.add_grain(out, norm, center + d, L, sig, t_out)
        t_out += T
        cyc += 1
    return out[:out_len] / np.maximum(norm[:out_len], 1e-6)


def v5b_timestretch(x, sr, alpha, fmin=70.0, fmax=500.0, hop=256):
    x = x.astype(np.float64)
    f0s, voiced = ps.robust_f0(x, sr, fmin, fmax, hop)
    per = ps.smooth_period(f0s, voiced, sr, fmin, fmax)
    pol = ps.excitation_polarity(x)
    marks = ps.analysis_marks(x, voiced, per, pol)
    if len(marks) < 3:
        return np.zeros(int(round(len(x) * alpha)))
    sos = butter(6, FC / (sr / 2), btype="low", output="sos")
    lo = sosfiltfilt(sos, x)
    hi = x - lo
    y_lo = _stretch_band(lo, marks, per, voiced, alpha, high_band=False)
    y_hi = _stretch_band(hi, marks, per, voiced, alpha, high_band=True)
    m = min(len(y_lo), len(y_hi))
    return y_lo[:m] + y_hi[:m]


def main():
    inp, outp, alpha = sys.argv[1], sys.argv[2], float(sys.argv[3])
    x, sr = sf.read(inp)
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = v5b_timestretch(x, sr, alpha)
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 1.0:
        y = y / peak * 0.99
    sf.write(outp, y.astype(np.float32), sr)
    print(f"[v5b] {inp} -> {outp}  {len(x)}->{len(y)} ({len(y)/sr:.3f}s) peak={peak:.3f}")


if __name__ == "__main__":
    main()
