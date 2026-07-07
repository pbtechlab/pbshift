#!/usr/bin/env python3
"""Reference-free 'chorusing/phasiness' metric for time-stretch output.

Chorusing in a phase-vocoder is loss of vertical/temporal phase coherence:
partials get slow amplitude beating (flutter) and their spectral lines broaden
into multiple detuned copies. This measures both on the STEADY interior of a
stretched signal, where the source itself is ~constant so any modulation is an
artifact.

Two components (higher = more chorusing):
  flutter_db : median over strong partials of the depth (dB std) of each
               partial's amplitude envelope in the 1-25 Hz modulation band.
  spread     : median normalized spectral spread around each partial peak
               (line broadening from detuned copies).
"""
import sys
import numpy as np
import soundfile as sf


def analytic_partials(x, sr):
    # STFT to find strong, stable partials in the steady interior
    n = len(x)
    lo, hi = int(0.2 * n), int(0.8 * n)  # interior, avoid edges/onset
    x = x[lo:hi]
    N = 8192
    hop = N // 4
    win = np.hanning(N)
    frames = []
    for s in range(0, len(x) - N, hop):
        frames.append(np.fft.rfft(x[s:s + N] * win))
    if len(frames) < 8:
        return None
    S = np.array(frames)                 # [T, F]
    mag = np.abs(S)
    avg = mag.mean(axis=0)
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    # pick peaks in the average spectrum
    peaks = []
    thr = avg.max() * 0.02
    for k in range(2, len(avg) - 2):
        if avg[k] > thr and avg[k] >= avg[k - 1] and avg[k] > avg[k + 1] \
           and avg[k] > avg[k - 2] and avg[k] > avg[k + 2]:
            peaks.append(k)
    peaks = sorted(peaks, key=lambda k: -avg[k])[:24]
    return S, mag, avg, freqs, peaks, hop, sr


def measure(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    r = analytic_partials(x, sr)
    if r is None:
        return None
    S, mag, avg, freqs, peaks, hop, sr = r
    frame_rate = sr / hop
    flutters = []
    spreads = []
    for k in peaks:
        env = mag[:, k]
        env = env / (env.mean() + 1e-12)
        # modulation spectrum of the envelope, 1..25 Hz band vs DC
        e = env - env.mean()
        E = np.abs(np.fft.rfft(e * np.hanning(len(e))))
        mfreqs = np.fft.rfftfreq(len(e), 1.0 / frame_rate)
        band = (mfreqs >= 1.0) & (mfreqs <= 25.0)
        if band.sum() > 0:
            modpow = np.sqrt((E[band] ** 2).mean())
            flutters.append(20 * np.log10(modpow + 1e-6))
        # spectral spread around the peak (line broadening)
        k0, k1 = max(0, k - 6), min(mag.shape[1], k + 7)
        loc = avg[k0:k1]
        loc = loc / (loc.sum() + 1e-12)
        cbins = np.arange(k0, k1)
        centroid = (loc * cbins).sum()
        spread = np.sqrt((loc * (cbins - centroid) ** 2).sum())
        spreads.append(spread)
    return {
        "flutter_db": float(np.median(flutters)) if flutters else float("nan"),
        "spread": float(np.median(spreads)) if spreads else float("nan"),
        "npeaks": len(peaks),
    }


if __name__ == "__main__":
    for p in sys.argv[1:]:
        m = measure(p)
        if m is None:
            print(f"{p}\tSKIP(too short)")
        else:
            print(f"{p}\tflutter_db={m['flutter_db']:+.2f}\t"
                  f"spread={m['spread']:.3f}\tpeaks={m['npeaks']}")
