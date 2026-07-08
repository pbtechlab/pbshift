#!/usr/bin/env python3
"""noise_probe_flat.py — a NOISE metric from the SPECTRAL-FLATNESS / BROADBAND-
FLOOR family, built to match the EAR on 2x voice time-stretch.

Motivation
----------
Our geometry-based probes (cepstral comb, group-delay dispersion, pre-echo,
pitch-sync crest) score the WI arms (PSMS / WITS) as GOOD, yet a human calls
them UNLISTENABLE broadband NOISE. Those arms normalize each pitch cycle to a
fixed length by resampling, blend, then un-normalize by resampling again, and
regenerate unvoiced spans as random-phase noise. The audible failure is a
raised BROADBAND NOISE FLOOR: energy poured into the deep inter-harmonic and
high-frequency spectral VALLEYS of the voice. Clean voice is peaky (a harmonic
comb over a formant envelope, with 30-50 dB nulls between the teeth). Noise
fills the nulls. So the ear's complaint is literally "the spectrum got flatter /
the valleys came up." This probe measures exactly that, several ways.

Ground-truth human ordering (cleanest -> worst), 2x stretch of
Sample/tda_0121_clm_n_f.wav:
    CLEAN        : input (reference),  Bungee
    LISTENABLE   : PSOLA               (slight flange/delay, but time-domain)
    UNLISTENABLE : PSMS,  WITS         (broadband noise)
The listenable PV competitors (Signalsmith, SoundTouch, RubberBand) must land
better than PSMS/WITS.

We want a single measure M with:  M(WITS),M(PSMS) > M(PSOLA) > M(Bungee)~=M(input)

Every measure below is in the spectral-flatness / broadband-floor family:
  * ltas_sfm_db        spectral flatness (geo/arith mean) of the long-term
                       average spectrum. Flat = noisy.
  * frame_sfm          per-frame spectral flatness averaged over ACTIVE frames
                       (captures the harmonic-comb nulls that LTAS smears out).
  * frame_sfm_hf       same, restricted to the HF hiss band.
  * spec_entropy       per-frame normalized spectral entropy (flat = high).
  * hf_ratio_db        high-frequency energy fraction (hiss), delta vs input.
  * floor_below_peak   LTAS noise floor (low percentile) below its peak; a low
                       value = shallow spectrum = raised floor = noisy.
  * valley_excess_db   THE headline valley measure: align each engine's LTAS to
                       the input's on the voice band, then average how many dB it
                       sits ABOVE the input inside the input's own spectral
                       valleys. Positive = added noise floor. (Task's request.)

usage: python benchmarks/noise_probe_flat.py
"""
import os
import sys
import numpy as np
import soundfile as sf
from scipy.ndimage import minimum_filter1d, uniform_filter1d

try:                                    # keep the em-dashes on a cp932 console
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EPS = 1e-12
SR = 44100

# ---- files (relative to repo root) -----------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [
    ("input",        "Sample/tda_0121_clm_n_f.wav"),        # CLEAN (reference)
    ("Bungee",       "Sample/tda_0121_2x_Bungee.wav"),      # CLEAN
    ("PSOLA",        "Sample/tda_0121_2x_pbshift_psola.wav"),  # LISTENABLE (mid)
    ("PSMS",         "Sample/tda_0121_2x_pbshift_psms.wav"),   # UNLISTENABLE
    ("WITS",         "Sample/tda_0121_2x_pbshift_wits.wav"),   # UNLISTENABLE
    ("Signalsmith",  "Sample/tda_0121_2x_Signalsmith.wav"),  # listenable PV
    ("SoundTouch",   "Sample/tda_0121_2x_SoundTouch.wav"),   # listenable PV
    ("RubberBand",   "Sample/tda_0121_2x_RubberBand.wav"),   # listenable PV
    ("PSMS_bl",      "benchmarks/_grid/psms_bl.wav"),        # bandlimited retry
    ("WITS_bl",      "benchmarks/_grid/wits_bl.wav"),        # bandlimited retry
]

# Human rank for the CANONICAL labelled set (used to check ordering).
# 0 = clean, 1 = listenable-mid, 2 = unlistenable-noise.
HUMAN_RANK = {"input": 0, "Bungee": 0, "PSOLA": 1, "PSMS": 2, "WITS": 2}

N_FFT = 4096
HOP = 1024
HF_LO = 5500.0          # hiss band lower edge (Hz)
VOICE_LO, VOICE_HI = 120.0, 1600.0   # LTAS alignment anchor band (fundamental+F1)


def load_mono(path):
    x, sr = sf.read(os.path.join(ROOT, path), dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        raise SystemExit(f"expected sr={SR}, got {sr} for {path}")
    return x


def stft_power(x):
    """Framed power spectrogram (frames x bins) with a Hann window."""
    w = np.hanning(N_FFT)
    n = len(x)
    if n < N_FFT:
        x = np.pad(x, (0, N_FFT - n))
        n = N_FFT
    starts = range(0, n - N_FFT + 1, HOP)
    P = np.empty((len(list(starts)), N_FFT // 2 + 1))
    for i, s in enumerate(range(0, n - N_FFT + 1, HOP)):
        seg = x[s:s + N_FFT] * w
        P[i] = np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    return P, freqs


def active_mask(P, thr_db=35.0):
    """Frames within thr_db of the loudest frame (drop leading/trailing silence
    and inter-word gaps whose spectra are meaningless / trivially flat)."""
    e = P.sum(axis=1)
    ref = e.max() + EPS
    return e >= ref * 10 ** (-thr_db / 10.0)


def flatness(p, axis=-1):
    """Spectral flatness = geometric mean / arithmetic mean (Wiener entropy).
    0 = pure tone / peaky, 1 = white / flat. p is power."""
    p = np.maximum(p, EPS)
    gm = np.exp(np.mean(np.log(p), axis=axis))
    am = np.mean(p, axis=axis)
    return gm / am


def spectral_entropy(p, axis=-1):
    """Normalized Shannon entropy of the (power) spectrum, in [0,1]. Flat = 1."""
    p = np.maximum(p, EPS)
    pn = p / p.sum(axis=axis, keepdims=True)
    H = -np.sum(pn * np.log(pn), axis=axis)
    return H / np.log(p.shape[axis])


def measures(x):
    P, freqs = stft_power(x)
    act = active_mask(P)
    Pa = P[act]                                   # active frames only
    ltas = Pa.mean(axis=0)                        # long-term avg spectrum (power)
    ltas_db = 10 * np.log10(ltas + EPS)

    hf = freqs >= HF_LO
    voice = (freqs >= VOICE_LO) & (freqs <= VOICE_HI)

    out = {}
    # 1. LTAS spectral flatness (full band and HF band), in dB
    out["ltas_sfm_db"] = 10 * np.log10(flatness(ltas) + EPS)
    out["ltas_sfm_hf_db"] = 10 * np.log10(flatness(ltas[hf]) + EPS)

    # 2. Per-frame spectral flatness averaged over active frames
    out["frame_sfm"] = float(np.mean(flatness(Pa, axis=1)))
    out["frame_sfm_hf"] = float(np.mean(flatness(Pa[:, hf], axis=1)))

    # 3. Per-frame spectral entropy (normalized), averaged
    out["spec_entropy"] = float(np.mean(spectral_entropy(Pa, axis=1)))

    # 4. HF energy fraction (hiss), in dB relative to total
    hf_frac = ltas[hf].sum() / (ltas.sum() + EPS)
    out["hf_ratio_db"] = 10 * np.log10(hf_frac + EPS)

    # 5. LTAS noise floor below peak (dB). Peaky (clean) -> large; raised -> small
    peak = np.percentile(ltas_db, 99.0)
    floor = np.percentile(ltas_db, 10.0)
    out["floor_below_peak"] = float(peak - floor)

    # keep raw spectra for cross-file valley comparison
    out["_ltas_db"] = ltas_db
    out["_freqs"] = freqs
    return out


def valley_excess(ref_db, eng_db, freqs, anchor):
    """Headline valley measure. Align the engine LTAS to the input LTAS by a
    single dB offset chosen on the voice anchor band (fundamental+F1, where the
    real signal lives and clean stretchers agree). Then find the INPUT's own
    spectral VALLEYS (bins that dip >=6 dB below the input's local upper
    envelope) and average how many dB the aligned engine sits ABOVE the input
    there. Positive => the engine poured a broadband noise floor into the nulls."""
    off = np.median(eng_db[anchor]) - np.median(ref_db[anchor])
    eng_a = eng_db - off
    # input's local upper envelope via a wide max-ish (use -min of -x) then smooth
    env = -minimum_filter1d(-ref_db, size=41)
    env = uniform_filter1d(env, size=21)
    valley = ref_db <= (env - 6.0)
    if valley.sum() < 8:
        valley = ref_db <= np.percentile(ref_db, 40.0)
    excess = eng_a[valley] - ref_db[valley]
    return float(np.mean(excess)), valley


def main():
    res = {name: measures(load_mono(path)) for name, path in FILES}
    freqs = res["input"]["_freqs"]
    anchor = (freqs >= VOICE_LO) & (freqs <= VOICE_HI)
    ref_db = res["input"]["_ltas_db"]

    # valley excess vs input for every file
    for name in res:
        ve, _ = valley_excess(ref_db, res[name]["_ltas_db"], freqs, anchor)
        res[name]["valley_excess_db"] = ve

    cols = ["ltas_sfm_db", "ltas_sfm_hf_db", "frame_sfm", "frame_sfm_hf",
            "spec_entropy", "hf_ratio_db", "floor_below_peak", "valley_excess_db"]

    # ---- table ----
    print("\n2x VOICE STRETCH — spectral-flatness / broadband-floor probe")
    print("higher = noisier for every column EXCEPT floor_below_peak "
          "(higher = cleaner)\n")
    hdr = f"{'file':13s} " + " ".join(f"{c:>16s}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    order = ["input", "Bungee", "PSOLA", "Signalsmith", "SoundTouch",
             "RubberBand", "PSMS", "WITS", "PSMS_bl", "WITS_bl"]
    for name in order:
        row = f"{name:13s} " + " ".join(f"{res[name][c]:16.4f}" for c in cols)
        print(row)

    # ---- delta vs input ----
    print("\nDELTA vs input (engine - input):")
    print(hdr)
    print("-" * len(hdr))
    for name in order:
        if name == "input":
            continue
        row = f"{name:13s} " + " ".join(
            f"{res[name][c] - res['input'][c]:16.4f}" for c in cols)
        print(row)

    # ---- ordering check on the canonical labelled set ----
    print("\nORDERING CHECK (canonical labelled set):")
    print("need  WITS,PSMS (rank2)  >  PSOLA (rank1)  >  Bungee~input (rank0)")
    print("[floor_below_peak is inverted before checking]\n")
    labelled = list(HUMAN_RANK.keys())
    best = None
    for c in cols:
        vals = {n: res[n][c] for n in labelled}
        if c == "floor_below_peak":              # invert so higher = noisier
            vals = {n: -v for n, v in vals.items()}
        # group means by human rank
        g0 = np.mean([vals[n] for n in labelled if HUMAN_RANK[n] == 0])
        g1 = np.mean([vals[n] for n in labelled if HUMAN_RANK[n] == 1])
        g2 = np.mean([vals[n] for n in labelled if HUMAN_RANK[n] == 2])
        mono = (g2 > g1 > g0)
        # strict pairwise: every rank2 > PSOLA > every rank0
        r2 = [vals["PSMS"], vals["WITS"]]
        r0 = [vals["input"], vals["Bungee"]]
        strict = (min(r2) > vals["PSOLA"] > max(r0))
        # also require the listenable PV competitors beat the noise arms
        pv_ok = all(res[p][c] < min(res["PSMS"][c], res["WITS"][c])
                    if c != "floor_below_peak"
                    else res[p][c] > max(res["PSMS"][c], res["WITS"][c])
                    for p in ["Signalsmith", "SoundTouch", "RubberBand"])
        # separation margin (rank2 mean - rank1) normalized by (rank1 - rank0)
        sep = (g2 - g1) / (abs(g1 - g0) + 1e-9)
        tag = "STRICT+PV" if (strict and pv_ok) else ("STRICT" if strict
              else ("group-mono" if mono else "-"))
        print(f"  {c:18s} g0={g0:9.4f} g1={g1:9.4f} g2={g2:9.4f}  "
              f"sep={sep:7.2f}  {tag}")
        if strict and pv_ok and (best is None or sep > best[1]):
            best = (c, sep)

    print()
    if best:
        print(f"BEST measure reproducing the human ordering: {best[0]}  "
              f"(strict + PV competitors, separation={best[1]:.2f})")
    else:
        print("No single measure gave STRICT+PV ordering; see group-mono tags.")


if __name__ == "__main__":
    main()
