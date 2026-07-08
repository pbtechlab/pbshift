#!/usr/bin/env python3
"""Harmonic-to-noise / aperiodicity probe for voice time-stretch renders.

WHY THIS EXISTS
---------------
Our objective metrics (cepstral-comb, group-delay dispersion, pre-echo,
pitch-sync crest, and the GLOBAL single-f0 hnr_db in metrics.py) all score
GOOD on renders the ear calls NOISY / UNLISTENABLE. This probe searches the
harmonic-to-noise / aperiodicity family for a measure that tracks the EAR.

Ground truth (2x voice stretch of tda_0121_clm_n_f.wav), cleanest -> worst:
   CLEAN        : input (ref), Bungee
   LISTENABLE   : PSOLA (slight flange/delay)   [also Signalsmith/SoundTouch/RubberBand]
   UNLISTENABLE : PSMS, WITS  (broadband NOISE)

THE TRAP (documented here on purpose)
-------------------------------------
The two unlistenable renders fail in OPPOSITE directions on classic measures:
  * WITS re-randomises residual phase  -> genuine aperiodic noise.
  * PSMS interpolates length-normalised cycles -> an OVER-REGULARISED comb: a
    harsh buzz whose harmonics are TOO clean.
So every measure that scores "how well do the harmonics fit / how tall are the
peaks over the adjacent trough" is FOOLED by PSMS and ranks it the CLEANEST of
all files:
    HNR_ac  (autocorrelation HNR)         PSMS highest  -> WRONG
    CPP     (cepstral peak prominence)     PSMS highest  -> WRONG
    Aper_ls (least-squares harmonic resid) PSMS lowest   -> WRONG
These are energy/peak-weighted: they are dominated by the strong harmonic peaks
and blind to the LOW-ENERGY, WIDE-BAND raised noise floor the ear actually hears.

THE FIX
-------
FloorFlat = spectral FLATNESS (Wiener entropy, geometric/arithmetic mean, in dB)
of ONLY the INTER-HARMONIC bins (everything not within +-10% of a k*f0 line).
Added broadband noise fills the deep spectral valleys between harmonics. In the
GEOMETRIC-MEAN (log) domain those near-zero valleys dominate: a clean voice has
60+ dB nulls (very peaky floor, FloorFlat ~ -21 dB); added noise raises the
nulls, compressing the range (FloorFlat ~ -15 dB). It measures the FLOOR, not
the PEAKS, so the over-regularised PSMS comb no longer hides its noise.
=> FloorFlat flags PSMS AND WITS as the two noisy outliers (~5-6 dB / ~4x above
   the listenable cluster). It is the member of the family that tracks the ear.

usage: python benchmarks/noise_probe_hnr.py            # runs the labelled set
       python benchmarks/noise_probe_hnr.py a.wav ...  # arbitrary files
"""
import sys
import os
import numpy as np
import soundfile as sf
import librosa

EPS = 1e-12
FMIN, FMAX = 70.0, 500.0        # voice f0 search range (Hz)
HOP = 256
FRAME = 4096                    # ~93 ms: resolves individual harmonics
HARM_BW = 0.10                  # harmonic half-width as fraction of f0
BAND = (70.0, 8000.0)           # analysis band (Hz)


def load_mono(path):
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def track_f0(x, sr):
    f0, vflag, _ = librosa.pyin(x, fmin=FMIN, fmax=FMAX, sr=sr,
                                frame_length=2048, hop_length=HOP)
    voiced = np.nan_to_num(vflag, nan=0.0).astype(bool) & np.isfinite(f0)
    return np.nan_to_num(f0, nan=0.0), voiced


# ---- classic family members (the TRAP: all fooled by the PSMS comb) --------
def hnr_autocorr(seg, sr):
    """Boersma-style autocorrelation HNR (dB). Peak-of-periodicity based."""
    L = len(seg)
    win = np.hanning(L)
    w = seg * win
    N = int(2 ** np.ceil(np.log2(2 * L)))
    ac = np.fft.irfft(np.abs(np.fft.rfft(w, N)) ** 2, N)[:L]
    acw = np.fft.irfft(np.abs(np.fft.rfft(win, N)) ** 2, N)[:L]
    r = ac / (acw + EPS)
    if r[0] <= 0:
        return None
    r = r / r[0]
    lo, hi = max(1, int(sr / FMAX)), min(L - 1, int(sr / FMIN))
    if hi <= lo + 1:
        return None
    R = float(np.clip(r[lo:hi].max(), 1e-6, 1 - 1e-6))
    return 10.0 * np.log10(R / (1.0 - R))


def cpp(seg, sr):
    """Cepstral Peak Prominence (dB). Rahmonic peak over cepstral regression."""
    L = len(seg)
    w = seg * np.hanning(L)
    N = int(2 ** np.ceil(np.log2(L)))
    logp = np.log(np.abs(np.fft.rfft(w, N)) ** 2 + EPS)
    ceps = np.fft.irfft(logp, N)
    cdb = 10.0 * np.log10(ceps ** 2 + EPS)
    q_lo, q_hi = max(2, int(sr / FMAX)), min(N // 2, int(sr / FMIN))
    if q_hi <= q_lo + 2:
        return None
    q = np.arange(q_lo, q_hi)
    A = np.vstack([q, np.ones_like(q)]).T
    slope, icept = np.linalg.lstsq(A, cdb[q_lo:q_hi], rcond=None)[0]
    return float((cdb[q_lo:q_hi] - (slope * q + icept)).max())


# ---- the FIX: inter-harmonic noise-floor flatness --------------------------
def floor_flatness(seg, sr, f0i):
    """Spectral flatness (dB) of the INTER-HARMONIC bins only. Higher (less
    negative) = raised broadband noise floor = NOISIER. Also returns the
    energy-ratio HtN (harmonic/floor, dB) which is the peak-weighted twin that
    FAILS, for contrast."""
    L = len(seg)
    S = np.abs(np.fft.rfft(seg * np.hanning(L))) ** 2
    df = sr / L
    K = len(S)
    fbin = np.arange(K) * df
    mask = np.zeros(K, bool)
    w = max(1, int(HARM_BW * f0i / df))
    kmax = int(min(BAND[1], sr / 2 - f0i) / f0i)
    for k in range(1, kmax + 1):
        c = int(round(k * f0i / df))
        mask[max(0, c - w):min(K, c + w + 1)] = True
    inband = (fbin >= BAND[0]) & (fbin <= BAND[1])
    floor = S[(~mask) & inband]
    harm = S[mask & inband]
    if len(floor) == 0 or len(harm) == 0:
        return None, None
    fl = floor + EPS
    ff = 10.0 * np.log10(np.exp(np.mean(np.log(fl))) / np.mean(fl))
    htn = 10.0 * np.log10(harm.sum() / (floor.sum() + EPS))   # peak-weighted (fails)
    return ff, htn


def probe(path):
    x, sr = load_mono(path)
    f0, voiced = track_f0(x, sr)
    n = len(x)
    acc = {k: [] for k in ("HNR_ac", "CPP", "HtN", "FloorFlat")}
    for i, v in enumerate(voiced):
        if not v or f0[i] < FMIN or f0[i] > FMAX:
            continue
        a = i * HOP - FRAME // 2
        b = a + FRAME
        if a < 0 or b > n:
            continue
        seg = x[a:b]
        if np.sqrt(np.mean(seg ** 2)) < 1e-4:
            continue
        h = hnr_autocorr(seg, sr)
        if h is not None:
            acc["HNR_ac"].append(h)
        c = cpp(seg, sr)
        if c is not None:
            acc["CPP"].append(c)
        ff, htn = floor_flatness(seg, sr, f0[i])
        if ff is not None:
            acc["FloorFlat"].append(ff)
            acc["HtN"].append(htn)
    med = lambda a: float(np.median(a)) if a else float("nan")
    out = {k: med(v) for k, v in acc.items()}
    out["n_voiced"] = len(acc["FloorFlat"])
    return out


LABELLED = [
    ("input",       0, "Sample/tda_0121_clm_n_f.wav"),
    ("Bungee",      0, "Sample/tda_0121_2x_Bungee.wav"),
    ("PSOLA",       1, "Sample/tda_0121_2x_pbshift_psola.wav"),
    ("PSMS",        2, "Sample/tda_0121_2x_pbshift_psms.wav"),
    ("WITS",        2, "Sample/tda_0121_2x_pbshift_wits.wav"),
    ("Signalsmith", 1, "Sample/tda_0121_2x_Signalsmith.wav"),
    ("SoundTouch",  1, "Sample/tda_0121_2x_SoundTouch.wav"),
    ("RubberBand",  1, "Sample/tda_0121_2x_RubberBand.wav"),
    ("psms_bl",    -1, "benchmarks/_grid/psms_bl.wav"),
    ("wits_bl",    -1, "benchmarks/_grid/wits_bl.wav"),
]

# columns: which reproduce the ear? FAIL = fooled by PSMS comb; WIN = tracks ear
COLS = [("HNR_ac", "FAIL hi=clean"), ("CPP", "FAIL hi=clean"),
        ("HtN", "FAIL hi=clean"), ("FloorFlat", "WIN  hi=NOISY")]


def main():
    rows = ([("", None, p) for p in sys.argv[1:]] if len(sys.argv) > 1
            else LABELLED)
    hdr = f"{'label':13s}{'grp':>4s}{'nV':>6s}" + "".join(
        f"{c:>12s}" for c, _ in COLS)
    print(hdr)
    print(f"{'':13s}{'':>4s}{'':>6s}" + "".join(f"{tag:>12s}" for _, tag in COLS))
    print("-" * len(hdr))
    for label, grp, path in rows:
        if not os.path.exists(path):
            print(f"{label:13s}  MISSING {path}")
            continue
        r = probe(path)
        g = "" if grp is None else str(grp)
        print(f"{label:13s}{g:>4s}{r['n_voiced']:>6d}" +
              "".join(f"{r[c]:>12.3f}" for c, _ in COLS))
    print("\ngrp: 0=CLEAN 1=LISTENABLE 2=UNLISTENABLE  -1=unlabelled retry")
    print("FloorFlat is the only column where the UNLISTENABLE pair (PSMS,WITS)")
    print("are the two NOISIEST (highest); the others rank PSMS as the cleanest.")


if __name__ == "__main__":
    main()
