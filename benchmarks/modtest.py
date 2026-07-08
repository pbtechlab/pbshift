#!/usr/bin/env python3
"""Chorus/doubler diagnostic on a PERFECTLY STEADY synthetic tone.

Synthesizes a dead-steady harmonic tone (constant amplitude, no vibrato), then
stretches it with each engine given on the command line and reports the output
amplitude-envelope MODULATION SPECTRUM (0.5-40 Hz). Source modulation is ~0 by
construction, so every dB of output wobble is pure time-stretch artifact —
exactly the 'chorus / modulation / doubler' the ear hears.

usage: modtest.py <engine-exe> <label> <fftHint> <hopHint> [stretch...]
Renders to benchmarks/chorus/mt_*.wav.  Fast (FFT only).
"""
import os
import subprocess
import sys
import numpy as np
import soundfile as sf

SR = 48000
OUT = "benchmarks/chorus"
TONE = f"{OUT}/mt_steady.wav"


def make_steady_tone(seconds=3.0, f0=220.0, nharm=6):
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for k in range(1, nharm + 1):
        x += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t + 0.3 * k)
    x *= 0.2 / np.max(np.abs(x))
    st = np.stack([x, x], axis=1).astype(np.float32)   # steady stereo
    sf.write(TONE, st, SR)


def mod_spectrum(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = len(x)
    x = x[int(0.2 * n):int(0.8 * n)]
    # analytic envelope
    X = np.fft.fft(x)
    h = np.zeros(len(x)); h[0] = 1
    h[1:(len(x) + 1) // 2] = 2
    if len(x) % 2 == 0:
        h[len(x) // 2] = 1
    env = np.abs(np.fft.ifft(X * h))
    ds = max(1, sr // 500)
    e = env[::ds]; esr = sr / ds
    e = e / (np.mean(e) + 1e-12); e = e - np.mean(e)
    E = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    f = np.fft.rfftfreq(len(e), 1 / esr)
    band = (f >= 0.5) & (f <= 40)
    fb, Eb = f[band], E[band]
    wob = 20 * np.log10(np.sqrt(np.mean(Eb ** 2)) + 1e-9)
    order = np.argsort(Eb)[::-1]
    peaks = []
    for i in order:
        if all(abs(fb[i] - p[0]) > 1.5 for p in peaks):
            peaks.append((float(fb[i]), 20 * np.log10(float(Eb[i]) + 1e-9)))
        if len(peaks) >= 3:
            break
    return wob, peaks


def render(engine, inp, outp, stretch, extra):
    cmd = [os.path.abspath(engine), inp, outp, "--stretch", str(stretch)] + extra
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    engine, label, fftH, hopH = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    stretches = [float(s) for s in sys.argv[5:]] or [2.0, 4.0]
    extra = []
    if "PBMR_EXTRA" in os.environ:
        extra = os.environ["PBMR_EXTRA"].split()
    if not os.path.exists(TONE):
        make_steady_tone()
    w0, _ = mod_spectrum(TONE)
    print(f"  source(steady)          wobble={w0:+6.1f}dB")
    for s in stretches:
        o = f"{OUT}/mt_{label}_{s}.wav"
        render(engine, TONE, o, s, extra)
        w, pk = mod_spectrum(o)
        fr = SR / hopH
        pks = "  ".join(f"{f:.1f}Hz({a:+.0f})" for f, a in pk)
        print(f"  {label:18s} x{s:<4} wobble={w:+6.1f}dB  frameRate={fr:.1f}Hz  peaks[{pks}]")
