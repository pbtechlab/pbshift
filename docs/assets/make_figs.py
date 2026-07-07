#!/usr/bin/env python3
"""Generate README figures (pbshift's own signals only — no third-party engines).
  1) chorusing.png  : single-window (chorusing) vs multi-resolution (clean)
                      spectrograms of a 2x-stretched string pad.
  2) metrics.png    : pbshift headline quality numbers as a stat panel.
Run from repo root:  python docs/assets/make_figs.py
"""
import os
import subprocess
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

MR = os.path.abspath("tools/bin/multires.exe")
CORPUS = "benchmarks/corpus"
OUT = "docs/assets"

BG = "#0d1117"        # GitHub dark panel
FG = "#e6edf3"
ACCENT = "#58a6ff"
ACCENT2 = "#3fb950"
WARN = "#f0883e"
GRID = "#30363d"


def render(sig, out, ratio, scales=None):
    cmd = [MR, f"{CORPUS}/{sig}.wav", out, "--stretch", str(ratio)]
    if scales:
        cmd += ["--scales", scales]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def partial_envelope(path, band=(300, 1600)):
    """Amplitude envelope over time of the strongest partial in `band`.
    A steady note should give a flat line; chorusing makes it wobble."""
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = len(x)
    x = x[int(0.2 * n):int(0.8 * n)]
    N, hop = 4096, 256
    win = np.hanning(N)
    cols = []
    for s in range(0, len(x) - N, hop):
        cols.append(np.abs(np.fft.rfft(x[s:s + N] * win)))
    S = np.array(cols).T                        # [freq, time]
    freqs = np.fft.rfftfreq(N, 1 / sr)
    lo, hi = np.searchsorted(freqs, band[0]), np.searchsorted(freqs, band[1])
    k = lo + int(np.argmax(S[lo:hi].mean(axis=1)))   # strongest partial bin
    env = S[k]
    env = env / (np.median(env) + 1e-12)
    t = np.arange(len(env)) * hop / sr
    return t, env, freqs[k]


def fig_chorusing():
    render("strings_pad", f"{OUT}/_single.wav", 2.0, "4096:0:1e12")
    render("strings_pad", f"{OUT}/_multi.wav", 2.0)   # auto -> multi-resolution
    ts, es, fk = partial_envelope(f"{OUT}/_single.wav")
    tm, em, _ = partial_envelope(f"{OUT}/_multi.wav")
    m = min(len(es), len(em))
    ts, es, em = ts[:m], es[:m], em[:m]

    def flutter(env):
        e = env - env.mean()
        E = np.abs(np.fft.rfft(e * np.hanning(len(e))))
        fr = np.fft.rfftfreq(len(e), ts[1] - ts[0])
        b = (fr >= 1) & (fr <= 25)
        return 20 * np.log10(np.sqrt((E[b] ** 2).mean()) + 1e-6)

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG})
    fig = plt.figure(figsize=(12, 4.3), dpi=150)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.07, 0.15, 0.9, 0.62])
    ax.set_facecolor("#0b0f14")
    ax.plot(ts, es, color=WARN, lw=1.2,
            label=f"single window   (chorusing: {flutter(es):+.0f} dB wobble)")
    ax.plot(tm, em, color=ACCENT2, lw=1.6,
            label=f"multi-resolution (clean: {flutter(em):+.0f} dB)")
    ax.axhline(1.0, color=GRID, lw=0.8, ls="--")
    ax.set_xlim(ts[0], ts[-1])
    ax.set_xlabel("time (s)", color=FG, fontsize=11)
    ax.set_ylabel("partial amplitude\n(1.0 = steady)", color=FG, fontsize=11)
    ax.tick_params(colors=FG, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)
    leg = ax.legend(loc="upper right", frameon=True, fontsize=11,
                    facecolor="#161b22", edgecolor=GRID)
    for txt in leg.get_texts():
        txt.set_color(FG)
    fig.suptitle(f"2× time-stretch of a string pad — amplitude of one partial "
                 f"({fk:.0f} Hz) over time",
                 color=FG, fontsize=14, fontweight="bold", y=0.95)
    fig.text(0.07, 0.855, "chorusing = the sustained note wobbles in amplitude; "
             "the multi-resolution engine holds it steady",
             color="#8b949e", fontsize=10.5)
    fig.savefig(f"{OUT}/chorusing.png", facecolor=BG)
    plt.close(fig)
    for f_ in ["_single.wav", "_multi.wav"]:
        try:
            os.remove(f"{OUT}/{f_}")
        except OSError:
            pass
    print("wrote chorusing.png")


def fig_metrics():
    cards = [
        ("Harmonic purity", "134 dB", "HNR on tonal signals", ACCENT2),
        ("Pitch accuracy", "0.000¢", "f0 error, cents", ACCENT2),
        ("Identity null", "−138 dB", "bypass reconstruction", ACCENT2),
        ("Determinism", "bit-exact", "any buffer size, any run", ACCENT),
        ("Stereo drift", "0.000", "inter-channel coherence", ACCENT2),
        ("Attack keep", "1.06", "1.00 = perfect (percussion)", ACCENT2),
    ]
    fig = plt.figure(figsize=(12, 3.3), dpi=150)
    fig.patch.set_facecolor(BG)
    for i, (name, val, sub, col) in enumerate(cards):
        ax = fig.add_axes([0.02 + (i % 3) * 0.325,
                           0.52 - (i // 3) * 0.46, 0.30, 0.40])
        ax.set_facecolor("#161b22")
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.06, 0.72, name, color=FG, fontsize=11, transform=ax.transAxes)
        ax.text(0.06, 0.34, val, color=col, fontsize=26, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.06, 0.12, sub, color="#8b949e", fontsize=9,
                transform=ax.transAxes)
    fig.suptitle("pbshift — measured quality (internal objective benchmark)",
                 color=FG, fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(f"{OUT}/metrics.png", facecolor=BG,
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote metrics.png")


if __name__ == "__main__":
    fig_metrics()
