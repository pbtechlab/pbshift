#!/usr/bin/env python3
"""Evaluate one stretched wav: chorusing (flutter), transient sharpness, and
band-energy integrity vs a source (to reject degenerate/lowpassed outputs).

usage: eval_one.py <stretched.wav> [source.wav]
prints: flutter_db=<..> spread=<..> transient=<..> [integrity=OK|DEGENERATE
        dlow=<dB> dmid=<dB> dhi=<dB>]
Lower flutter = less chorusing. Higher transient = crisper attacks.
integrity compares band energies (source vs output); a band collapsing by
>20 dB relative to the others flags a broken (band-dropped) output.
"""
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, "benchmarks")
from phasiness import measure  # noqa: E402


def transient_sharp(x):
    e = np.abs(x)
    env = np.convolve(e, np.ones(32) / 32, "same")
    d = np.diff(env)
    d[d < 0] = 0
    return float(d.max() / (d.mean() + 1e-12))


def band_energies(x, sr):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    out = []
    for lo, hi in [(0, 800), (800, 5000), (5000, sr / 2)]:
        m = (f >= lo) & (f < hi)
        out.append(10 * np.log10((X[m] ** 2).sum() + 1e-12))
    return out  # [low, mid, hi] dB


def load(p):
    x, sr = sf.read(p)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def main():
    x, sr = load(sys.argv[1])
    m = measure(sys.argv[1]) or {"flutter_db": float("nan"), "spread": float("nan")}
    parts = [
        f"flutter_db={m['flutter_db']:+.2f}",
        f"spread={m['spread']:.3f}",
        f"transient={transient_sharp(x):.1f}",
    ]
    if len(sys.argv) > 2:
        s, ssr = load(sys.argv[2])
        sb = band_energies(s, ssr)
        ob = band_energies(x, sr)
        # normalize by low band (present in both), compare mid/hi retention
        dmid = (ob[1] - ob[0]) - (sb[1] - sb[0])
        dhi = (ob[2] - ob[0]) - (sb[2] - sb[0])
        degen = dmid < -20 or dhi < -20
        parts.append(f"integrity={'DEGENERATE' if degen else 'OK'}")
        parts.append(f"dmid={dmid:+.1f} dhi={dhi:+.1f}")
    print(" ".join(parts))


if __name__ == "__main__":
    main()
