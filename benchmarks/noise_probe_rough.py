#!/usr/bin/env python3
"""NOISE / ROUGHNESS probe — an objective measure that matches the EAR.

Our old metrics (cepstral comb, group-delay dispersion, pre-echo, pitch-sync
crest) score the broadband-NOISE renders (WITS/PSMS) as GOOD while the ear calls
them UNLISTENABLE. Those metrics look at PHASE/COMB structure. But the audible
defect in WITS/PSMS is not a comb — it is BROADBAND NOISE: the per-cycle
random-phase resynthesis + normalize/un-normalize resample round-trip FILLS THE
SPECTRAL VALLEYS BETWEEN HARMONICS with noise. Clean voice keeps deep valleys
(resolved harmonic lines) even up at 2-10 kHz; a noisy resynthesis flattens them.

WINNER
------
SFM_hi = voiced-gated HIGH-BAND (2-10 kHz) SPECTRAL FLATNESS (geo/arith mean of
the magnitude spectrum), median over voiced frames, measured with a
harmonic-RESOLVING window (nperseg=2048 -> ~21 Hz bins, well below this voice's
~200 Hz F0 so the inter-harmonic valleys are resolved). This is the spectral-
kurtosis / peakiness family: flatness -> 1 means noise-like (flat, valleys
filled); flatness -> 0 means tonal (sharp harmonic lines, deep valleys).

It is RATE-ROBUST (a per-frame spectral-shape statistic, so the 1x input and the
2x renders are directly comparable — unlike any temporal flux/roughness rate
metric, which the 1x natural voice defeats) and VOICE-GATED (scored only on
strong, tonal frames, so natural fricatives don't count as artifact).

Result on the labelled set (higher = noisier):
    input  0.344   Bungee 0.326   |  PSOLA 0.353  |  PSMS 0.456   WITS 0.540
    ------- CLEAN -------            -- LISTEN --     ---- UNLISTENABLE ----
Reproduces the full human ordering:  input,Bungee < PSOLA < PSMS,WITS.
All listenable PV competitors (Signalsmith/SoundTouch/RubberBand) land in the
CLEAN band; the bandlimited retries (_grid/*_bl) stay in the NOISE band.

The other columns are the rest of the roughness/modulation/kurtosis family that
were TRIED and do NOT reproduce the ordering (see REPORT at bottom): per-band AM
ROUGHNESS collapses because PSMS's voiced path is interpolated (smooth, not
rough); temporal FLICKER/HF-flux flags PSMS/WITS but the 1x input's natural
frication reads as noisy (rate confound); CREST does not separate cleanly.

usage: python benchmarks/noise_probe_rough.py            # runs the labelled set
       python benchmarks/noise_probe_rough.py a.wav ...  # custom files
"""
import sys
import numpy as np
import soundfile as sf
from scipy.signal import stft, butter, sosfiltfilt, hilbert

# ---- labelled ground-truth set (human ordering, best -> worst) --------------
LABELLED = [
    ("input   (CLEAN ref)",  "Sample/tda_0121_clm_n_f.wav",          "clean"),
    ("Bungee  (CLEAN)",      "Sample/tda_0121_2x_Bungee.wav",        "clean"),
    ("PSOLA   (LISTEN/mid)", "Sample/tda_0121_2x_pbshift_psola.wav", "mid"),
    ("PSMS    (UNLISTEN)",   "Sample/tda_0121_2x_pbshift_psms.wav",  "noise"),
    ("WITS    (UNLISTEN)",   "Sample/tda_0121_2x_pbshift_wits.wav",  "noise"),
    ("Signalsmith (listen)", "Sample/tda_0121_2x_Signalsmith.wav",   "listen"),
    ("SoundTouch  (listen)", "Sample/tda_0121_2x_SoundTouch.wav",    "listen"),
    ("RubberBand  (listen)", "Sample/tda_0121_2x_RubberBand.wav",    "listen"),
    ("psms_bl (retry)",      "benchmarks/_grid/psms_bl.wav",         "?"),
    ("wits_bl (retry)",      "benchmarks/_grid/wits_bl.wav",         "?"),
]

NPERSEG = 2048          # harmonic-resolving window (~21 Hz bins @ 44.1k)
HI_LO, HI_HI = 2000.0, 10000.0   # high band where clean voice is quiet+tonal
ROUGH_LO, ROUGH_HI = 30.0, 150.0  # AM roughness modulation band (Hz)


def load(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    n = len(x)
    x = x[int(0.10 * n):int(0.90 * n)]        # steady interior
    return x / (np.max(np.abs(x)) + 1e-12), sr


def voiced_mask(f, S):
    """Strong, TONAL frames (real voiced speech), not silence or fricatives."""
    lowm = (f >= 80) & (f < 1500)
    Elow = S[lowm].sum(0)
    flat_low = np.exp(np.mean(np.log(S[lowm]), 0)) / (np.mean(S[lowm], 0) + 1e-12)
    return (Elow > np.percentile(Elow, 55)) & (flat_low < np.percentile(flat_low, 55))


# ---------- WINNER: voiced-gated high-band spectral flatness -----------------
def sfm_hi(x, sr):
    f, t, Z = stft(x, fs=sr, nperseg=NPERSEG, noverlap=NPERSEG * 3 // 4,
                   boundary=None)
    S = np.abs(Z) + 1e-8
    v = voiced_mask(f, S)
    m = (f >= HI_LO) & (f < HI_HI)
    Sh = S[m]
    flat = np.exp(np.mean(np.log(Sh), 0)) / (np.mean(Sh, 0) + 1e-12)
    return float(np.median(flat[v]))


# ---------- family measures that were TRIED (do not reproduce ordering) ------
def hf_flux(x, sr, lo=4000.0, hi=12000.0):
    """Temporal HF spectral flux (flicker). Flags PSMS/WITS but 1x input reads
    noisy too (rate confound: natural fast frication)."""
    f, t, Z = stft(x, fs=sr, nperseg=1024, noverlap=768, boundary=None)
    S = np.abs(Z) + 1e-8
    m = (f >= lo) & (f < hi)
    L = 20 * np.log10(S[m].sum(0) + 1e-6)
    return float(np.median(np.abs(np.diff(L))))


def am_roughness(x, sr):
    """Loudness-weighted critical-band AM roughness (30-150 Hz). Collapses here
    because PSMS's voiced path is smooth interpolation, not per-band AM."""
    edges = np.geomspace(120.0, min(0.45 * sr, 10000.0), 25)
    ds = max(1, int(sr // 1500)); esr = sr / ds
    rs, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sos = butter(4, np.clip([lo, hi], 1, sr / 2 - 1) / (sr / 2),
                     btype="band", output="sos")
        b = sosfiltfilt(sos, x)
        env = np.abs(hilbert(b))[::ds]
        e = env / (np.mean(env) + 1e-12); e = e - np.mean(e)
        E = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
        ff = np.fft.rfftfreq(len(e), 1 / esr)
        ac = ff > 1.0; band = (ff >= ROUGH_LO) & (ff <= ROUGH_HI)
        rs.append(E[band].sum() / (E[ac].sum() + 1e-20))
        ws.append(np.mean(b ** 2))
    ws = np.array(ws) / (np.sum(ws) + 1e-20)
    return float(np.sum(np.array(rs) * ws))


def crest_hi(x, sr):
    """Median voiced-frame high-band spectral crest (max/mean across bins)."""
    f, t, Z = stft(x, fs=sr, nperseg=NPERSEG, noverlap=NPERSEG // 2,
                   boundary=None)
    S = np.abs(Z) + 1e-8
    v = voiced_mask(f, S)
    m = (f >= HI_LO) & (f < HI_HI)
    Sh = S[m]
    crest = Sh.max(0) / (Sh.mean(0) + 1e-12)
    return float(np.median(crest[v]))


def main():
    files = LABELLED
    if len(sys.argv) > 1:
        files = [(p, p, "?") for p in sys.argv[1:]]

    print(f"{'file':22s} {'SFM_hi*':>8s} {'HFflux':>7s} {'AMrough':>8s} "
          f"{'crestHi':>8s}   group")
    print("-" * 70)
    vals = {}
    for label, path, grp in files:
        try:
            x, sr = load(path)
        except Exception as e:
            print(f"{label:22s} ERR {e}")
            continue
        s = sfm_hi(x, sr)
        vals[label.split()[0]] = s
        print(f"{label:22s} {s:8.4f} {hf_flux(x, sr):7.3f} "
              f"{am_roughness(x, sr):8.4f} {crest_hi(x, sr):8.2f}   [{grp}]")
    print("-" * 70)
    print("* SFM_hi = voiced-gated 2-10 kHz spectral flatness (WINNER). "
          "higher = noisier.")

    # verdict on the WINNER
    need = ("input", "Bungee", "PSOLA", "PSMS", "WITS")
    if all(k in vals for k in need):
        strict = (max(vals["input"], vals["Bungee"]) < vals["PSOLA"]
                  < min(vals["PSMS"], vals["WITS"]))
        twotier = (max(vals["input"], vals["Bungee"], vals["PSOLA"])
                   < min(vals["PSMS"], vals["WITS"]))
        print(f"\nSFM_hi verdict:  clean<PSOLA<noise (strict 3-tier) = {strict}")
        print(f"                 all-listenable < PSMS,WITS  (2-tier)  = {twotier}")


if __name__ == "__main__":
    main()
