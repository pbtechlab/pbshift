#!/usr/bin/env python3
"""Identity-locked phase-vocoder time-stretch (for STABLE voiced regions only).

On a stable vowel the spectrum is steady and there are no transients, so a
phase vocoder is essentially artifact-free there (its phasiness/comb problems
live in dynamic/broadband content). Crucially, a PV does NOT duplicate pitch
periods, so it has NONE of the frozen-duplicate "doubling" that PSOLA leaves on
sustained vowels (where consecutive periods are ~identical, corr ~0.99). Used
content-adaptively: PV where the signal is stable, PSOLA/WSOLA grains elsewhere.

Laroche-Dolson identity phase locking: peak-region rigid phase rotation keeps the
intra-partial phase structure, avoiding the classic PV smear.
"""
import numpy as np


def _princarg(p):
    return p - 2 * np.pi * np.round(p / (2 * np.pi))


def pv_stretch(x, alpha, N=2048, ola=4):
    """Identity-locked PV time-stretch of x by alpha. Returns length ~ len(x)*alpha."""
    x = np.asarray(x, dtype=np.float64)
    Ha = N // ola
    Hs = int(round(Ha * alpha))
    win = np.hanning(N + 1)[:N]
    nf = 1 + max(0, (len(x) - N) // Ha)
    out_len = (nf - 1) * Hs + N
    out = np.zeros(out_len + N)
    norm = np.zeros_like(out)
    bins = N // 2 + 1
    omega = 2 * np.pi * np.arange(bins) * Ha / N          # expected per-hop phase
    sum_phase = np.zeros(bins)
    prev_phase = np.zeros(bins)
    for i in range(nf):
        a = i * Ha
        frame = x[a:a + N]
        if len(frame) < N:
            frame = np.pad(frame, (0, N - len(frame)))
        S = np.fft.rfft(frame * win)
        mag = np.abs(S)
        phase = np.angle(S)
        if i == 0:
            sum_phase = phase.copy()
        else:
            dphi = _princarg(phase - prev_phase - omega)
            true_freq = omega + dphi
            sum_phase = sum_phase + (Hs / Ha) * true_freq
            # identity phase lock: rigidly rotate each magnitude-peak region by
            # the peak bin's propagated-minus-measured phase (preserves shape).
            peaks = _peaks(mag)
            if len(peaks):
                lo = 0
                locked = phase.copy()
                for k, p in enumerate(peaks):
                    hi = bins if k + 1 >= len(peaks) else \
                        (p + int(np.argmin(mag[p:peaks[k + 1] + 1])) + 1)
                    delta = _princarg(sum_phase[p] - phase[p])
                    locked[lo:hi] = phase[lo:hi] + delta
                    lo = hi
                sum_phase = locked
        prev_phase = phase
        of = np.fft.irfft(mag * np.exp(1j * sum_phase), N) * win
        out[i * Hs:i * Hs + N] += of
        norm[i * Hs:i * Hs + N] += win * win
    return out[:out_len] / np.maximum(norm[:out_len], 1e-6)


def _peaks(mag):
    tol = 1e-3 * mag.max()
    p = np.where((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:]) &
                 (mag[1:-1] > tol))[0] + 1
    return p
