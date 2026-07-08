#!/usr/bin/env python3
"""Noise gate: valley_excess_db for candidate renders vs the input.

The one metric that matched the human ear (clean ~0 dB, TD-PSOLA +1.3 dB
listenable, the failed WI renders +8..+12 dB unlistenable). Use as a hard gate
BEFORE asking the human to listen: a candidate must sit near/below PSOLA's level.

usage: python benchmarks/gate.py <wav> [<wav> ...]
       (input reference is Sample/tda_0121_clm_n_f.wav)
"""
import os
import sys
import numpy as np
import noise_probe_flat as nf
import echo_probe as ep
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

INPUT = "Sample/tda_0121_clm_n_f.wav"
CHAMP = "Sample/tda_0121_2x_pbshift_psola4.wav"  # v4 = current champion to beat
# Pitch 'n Time Pro (commercial vocal-mode 2x) — the ultimate reference target.
PNT = "Sample/tda_0121_2x_PnTPro.wav"


def band_comb(path, lo, hi):
    """Cepstral comb energy in a frequency band — 'doubler feel' tracks the HIGH
    band + smoothness better than broadband comb (v4 beats PnTPro broadband yet
    loses perceptually; PnTPro wins on hiComb 0.0070 + burst -3.62)."""
    x, sr = _load(path)
    sos = butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band", output="sos")
    xb = sosfiltfilt(sos, x)
    cav = ep.frame_avg_cepstrum(xb, sr)
    ci = ep.frame_avg_cepstrum(_INPUT_X[0], sr)
    pm, _ = ep.peak_in_band(ci, sr, 3.0, 11.0)
    _, v = ep.peak_in_band(cav, sr, 1.5, 25.0, exclude=[pm, 2 * pm, 0.5 * pm], guard_ms=0.8)
    return v


def _load(path):
    p = os.path.join(nf.ROOT, path) if os.path.exists(os.path.join(nf.ROOT, path)) else path
    x, sr = sf.read(p)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), sr


def comb_energy(path):
    """Flange detector (valley_excess can't see combs). Non-pitch cepstral comb
    energy in the 1.5-25 ms band, pitch quefrency masked. Lower = less flange."""
    x, sr = sf.read(os.path.join(nf.ROOT, path) if os.path.exists(
        os.path.join(nf.ROOT, path)) else path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    cav = ep.frame_avg_cepstrum(x, sr)
    ci = ep.frame_avg_cepstrum(_INPUT_X[0], sr)
    pm, _ = ep.peak_in_band(ci, sr, 3.0, 11.0)
    pex = [pm, 2 * pm, 0.5 * pm]
    _, v = ep.peak_in_band(cav, sr, 1.5, 25.0, exclude=pex, guard_ms=0.8)
    return v


_INPUT_X = [None]


def burst_index(x):
    """Momentary-noise detector the LTAS-based valley_excess misses: per-frame
    spectral flatness, 99.5th percentile (dB) over active frames. A brief noise
    burst flattens a few frames far above the rest, spiking the high percentile."""
    P, _ = nf.stft_power(x)
    Pa = P[nf.active_mask(P)]
    if len(Pa) < 4:
        return 0.0
    fl = 10.0 * np.log10(nf.flatness(Pa, axis=1) + nf.EPS)
    return float(np.percentile(fl, 99.5))


def main():
    cands = sys.argv[1:] or ["Sample/tda_0121_2x_pbshift_psola.wav"]
    files = [("input", INPUT)] + [(os.path.basename(p), p) for p in cands]
    res = {}
    for name, p in files:
        # allow paths relative to repo root or given as-is
        rel = p if os.path.exists(os.path.join(nf.ROOT, p)) else os.path.relpath(
            os.path.abspath(p), nf.ROOT)
        x = nf.load_mono(rel)
        res[name] = nf.measures(x)
        res[name]["_x"] = x
        res[name]["_burst"] = burst_index(x)
    freqs = res["input"]["_freqs"]
    anchor = (freqs >= nf.VOICE_LO) & (freqs <= nf.VOICE_HI)
    ref_db = res["input"]["_ltas_db"]
    inp_burst = burst_index(res["input"]["_x"])
    _INPUT_X[0] = res["input"]["_x"]
    champ_comb = comb_energy(CHAMP)     # v4 broadband comb baseline
    # append the PnT Pro reference row if available (the target to beat)
    show = list(files)
    if os.path.exists(os.path.join(nf.ROOT, PNT)):
        show.append(("PnTPro(target)", PNT))
    print(f"v4 champion comb={champ_comb:.4f}. 'doubler feel' tracks hiComb + burst "
          f"(PnTPro target: hiComb~0.007, burst~-3.6, valley~0).\n")
    print(f"{'file':32} {'valley':>8} {'burst':>7} {'comb':>7} {'hiComb':>7}  verdict")
    for name, p in show:
        m = res.get(name) or nf.measures(nf.load_mono(
            p if os.path.exists(os.path.join(nf.ROOT, p)) else os.path.relpath(
                os.path.abspath(p), nf.ROOT)))
        ve, _ = nf.valley_excess(ref_db, m["_ltas_db"], freqs, anchor)
        bi = burst_index(m["_x"]) - inp_burst if "_x" in m else burst_index(
            nf.load_mono(p if os.path.exists(os.path.join(nf.ROOT, p)) else
                         os.path.relpath(os.path.abspath(p), nf.ROOT))) - inp_burst
        cb = comb_energy(p)
        hc = band_comb(p, 2000.0, 6000.0)
        v_noise = abs(ve) <= 1.5
        v_burst = bi <= 1.2
        v_comb = cb <= champ_comb + 1e-4
        if name == "input":
            verdict = "CLEAN(ref)"
        elif "PnT" in name:
            verdict = "*** TARGET ***"
        elif not v_noise:
            verdict = "NOISY-reject"
        elif not v_burst:
            verdict = "BURST-reject"
        elif not v_comb:
            verdict = "no-comb-gain"
        else:
            verdict = "PASS (clean)"
        print(f"{name:32} {ve:8.2f} {bi:7.2f} {cb:7.4f} {hc:7.4f}  {verdict}")


if __name__ == "__main__":
    main()
