#!/usr/bin/env python3
"""Diagnose the 'chorus / doubler / modulation' artifact directly.

A cleanly stretched STEADY tone must stay dead steady. Any amplitude/phase
wobble in the output is 100% artifact. This measures:
  - the amplitude-envelope MODULATION SPECTRUM (0.5-40 Hz): where the wobble
    lives and how strong it is (the audible chorus/tremolo).
  - a DOUBLER index: short-lag normalized autocorrelation of the residual
    (a delayed/detuned copy shows up as an echo peak).
It marks the synthesis frame rate (sr / hop) so we can tell frame-rate
modulation (fixable with more overlap) from genuine phasiness.

usage: modspec.py <wav> [label] [--fftN N --hop H]   (N,H just annotate frame rate)
"""
import sys
import numpy as np
import soundfile as sf


def env_mod_spectrum(x, sr):
    # broadband amplitude envelope over the steady interior
    n = len(x)
    x = x[int(0.2 * n):int(0.8 * n)]
    # analytic-signal magnitude via FFT Hilbert
    X = np.fft.fft(x)
    h = np.zeros(len(x))
    h[0] = 1
    h[1:(len(x) + 1) // 2] = 2
    if len(x) % 2 == 0:
        h[len(x) // 2] = 1
    env = np.abs(np.fft.ifft(X * h))
    # downsample envelope to ~500 Hz for a clean low-freq mod spectrum
    ds = max(1, sr // 500)
    e = env[::ds]
    esr = sr / ds
    e = e / (np.mean(e) + 1e-12)
    e = e - np.mean(e)
    E = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    f = np.fft.rfftfreq(len(e), 1 / esr)
    band = (f >= 0.5) & (f <= 40)
    fb, Eb = f[band], E[band]
    # top 3 modulation peaks
    order = np.argsort(Eb)[::-1]
    peaks = []
    for i in order:
        if all(abs(fb[i] - p[0]) > 1.0 for p in peaks):
            peaks.append((float(fb[i]), float(Eb[i])))
        if len(peaks) >= 3:
            break
    # overall wobble energy in-band (RMS, dB rel to a tiny floor)
    wob = 20 * np.log10(np.sqrt(np.mean(Eb ** 2)) + 1e-9)
    return peaks, wob


def doubler_index(x, sr):
    # residual after removing slow trend; short-lag autocorr echo (2-60 ms)
    n = len(x)
    x = x[int(0.2 * n):int(0.8 * n)].astype(float)
    x = x - np.mean(x)
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(0.002 * sr), int(0.060 * sr)
    return float(np.max(np.abs(ac[lo:hi])))


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else path
    fftN = hop = None
    if "--fftN" in sys.argv:
        fftN = int(sys.argv[sys.argv.index("--fftN") + 1])
    if "--hop" in sys.argv:
        hop = int(sys.argv[sys.argv.index("--hop") + 1])
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peaks, wob = env_mod_spectrum(x, sr)
    di = doubler_index(x, sr)
    fr = f"  frameRate={sr/hop:.1f}Hz" if (fftN and hop) else ""
    pk = "  ".join(f"{f:.1f}Hz({20*np.log10(a+1e-9):+.0f}dB)" for f, a in peaks)
    print(f"{label:28s} wobble={wob:+6.1f}dB  doubler={di:.3f}  peaks[{pk}]{fr}")


if __name__ == "__main__":
    main()
