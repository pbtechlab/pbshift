# -*- coding: utf-8 -*-
"""
Objective quality metrics for time-scale modification / pitch shifting.

No time-aligned reference exists for TSM output, so we measure
artifact-specific quantities (following TSM-eval literature):

  duration_err      : output length vs expected (stretch * input)
  onset_recall      : fraction of expected onsets present (miss = smearing)
  onset_precision   : 1 - spurious onsets (doubling / stuttering)
  attack_ratio      : output/input 10-90% rise time at matched onsets (>1 = smeared)
  hnr_db            : harmonic-to-interharmonic-noise ratio for tonal signals
                      (drop vs input = phasiness / metallic noise)
  f0_err_cents      : measured output f0 vs expected f0
  envelope_lsd_db   : log-spectral distance of spectral envelopes on aligned
                      frames (formant preservation quality)
  ltas_dist_db      : long-term average spectrum distance (stretch-only)
  warble_db         : spurious 2-16 Hz modulation energy on steady tones
  stereo_coh_drift  : loss of interchannel coherence vs input
"""
import numpy as np
import soundfile as sf
from scipy.signal import stft, find_peaks, coherence

EPS = 1e-12


def load(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    return x.T, sr  # (ch, n)


def mono(x):
    return x.mean(axis=0)


# --------------------------------------------------------------- onsets
def onset_envelope(x, sr, n_fft=2048, hop=None):
    hop = hop or n_fft // 8
    f, t, Z = stft(x, sr, nperseg=n_fft, noverlap=n_fft - hop, padded=False)
    mag = np.abs(Z)
    logmag = np.log1p(1000 * mag)
    flux = np.diff(logmag, axis=1)
    flux[flux < 0] = 0
    env = flux.sum(axis=0)
    env = env / (np.median(env) + np.percentile(env, 95) + EPS)
    return env, hop / sr, t[1:]


def detect_onsets(x, sr):
    env, dt, times = onset_envelope(x, sr)
    thr = 0.12 + 1.2 * np.median(env)
    peaks, props = find_peaks(env, height=thr, distance=int(0.05 / dt))
    # absolute-level gate: reject "onsets" in near-silence (processing tails
    # 50+ dB below the signal's loud reference are inaudible, not events)
    k = max(1, int(0.005 * sr))
    rms = np.sqrt(np.convolve(x ** 2, np.ones(k) / k, mode="same") + EPS)
    ref = rms.max()
    keep = []
    for p in peaks:
        i = int(times[p] * sr)
        lo, hi = max(0, i - int(0.01 * sr)), min(len(x), i + int(0.02 * sr))
        if len(rms[lo:hi]) and rms[lo:hi].max() > ref * 10 ** (-50 / 20):
            keep.append(p)
    peaks = np.array(keep, dtype=int)
    return times[peaks], env, dt


def attack_rise_time(x, sr, onset_t, win=0.06):
    """10-90% rise time of local amplitude envelope around an onset."""
    i0 = max(0, int((onset_t - 0.02) * sr))
    i1 = min(len(x), int((onset_t + win) * sr))
    seg = np.abs(x[i0:i1])
    if len(seg) < 16:
        return None
    k = max(1, int(0.002 * sr))
    env = np.convolve(seg, np.ones(k) / k, mode="same")
    pk = env.max()
    if pk < 1e-5:
        return None
    lo = np.argmax(env > 0.1 * pk)
    hi = np.argmax(env > 0.9 * pk)
    if hi <= lo:
        return None
    return (hi - lo) / sr


def _match(t_ref, t_out, tol):
    """Greedy nearest matching; returns (hits, matched_out_mask, pairs)."""
    matched_out = np.zeros(len(t_out), bool)
    hits = 0
    pairs = []
    for te in t_ref:
        if len(t_out) == 0:
            break
        d = np.abs(t_out - te)
        j = int(np.argmin(d))
        if d[j] < tol and not matched_out[j]:
            matched_out[j] = True
            hits += 1
            pairs.append((te, t_out[j]))
    return hits, matched_out, pairs


def onset_metrics(x_in, y_out, sr_in, sr_out, stretch, truth_onsets=None):
    """recall + attack sharpness vs ground truth when available (valid even
    if truth is a partial-but-real event list); precision always vs the
    DETECTOR's own input events (a partial truth list would turn correctly
    reproduced events into fake false-positives)."""
    t_det_in, _, _ = detect_onsets(x_in, sr_in)
    t_out, _, _ = detect_onsets(y_out, sr_out)
    tol = 0.08 * max(1.0, stretch)

    res = {}
    ref = (np.asarray(truth_onsets, dtype=float)
           if truth_onsets is not None and len(truth_onsets) else t_det_in)
    if len(ref) == 0:
        return res
    hits, _, pairs = _match(ref * stretch, t_out, tol)
    res["onset_recall"] = hits / len(ref)
    ratios = []
    for te, to in pairs:
        r_in = attack_rise_time(x_in, sr_in, te / stretch)
        r_out = attack_rise_time(y_out, sr_out, to)
        if r_in and r_out:
            ratios.append(r_out / r_in)
    if ratios:
        res["attack_ratio"] = float(np.median(ratios))

    if len(t_det_in):
        hits_p, matched, _ = _match(t_det_in * stretch, t_out, tol)
        res["onset_precision"] = matched.sum() / max(len(t_out), 1)
    return res


# --------------------------------------------------------------- tonal
def measure_f0_fft(x, sr, fmin=40, fmax=2000):
    """f0 via FFT peak + harmonic product spectrum on middle section."""
    n = len(x)
    seg = x[n // 4: n // 4 + min(n // 2, 4 * sr)]
    seg = seg * np.hanning(len(seg))
    N = int(2 ** np.ceil(np.log2(len(seg)) + 1))
    S = np.abs(np.fft.rfft(seg, N))
    freqs = np.fft.rfftfreq(N, 1 / sr)
    hps = S.copy()
    for h in (2, 3, 4):
        dec = S[::h]
        hps[: len(dec)] *= dec
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        return None
    idx = np.argmax(hps[: len(freqs)][band])
    f0_coarse = freqs[band][idx]
    # refine on fundamental peak
    lo = int(f0_coarse * 0.9 * N / sr)
    hi = int(f0_coarse * 1.1 * N / sr) + 1
    j = lo + np.argmax(S[lo:hi])
    if 0 < j < len(S) - 1:  # parabolic interp
        a, b, c = np.log(S[j - 1] + EPS), np.log(S[j] + EPS), np.log(S[j + 1] + EPS)
        d = 0.5 * (a - c) / (a - 2 * b + c + EPS)
        return (j + d) * sr / N
    return f0_coarse


def hnr_db(x, sr, f0):
    """Harmonic energy vs inter-harmonic energy, middle 2s, long FFT."""
    n = len(x)
    seg = x[n // 4: n // 4 + min(n // 2, 2 * sr)]
    seg = seg * np.hanning(len(seg))
    N = int(2 ** np.ceil(np.log2(len(seg))))
    S = np.abs(np.fft.rfft(seg, N)) ** 2
    df = sr / N
    harm, inter = 0.0, 0.0
    k = 1
    while k * f0 < min(sr / 2 - f0, 16000):
        fk = k * f0
        w_h = max(2, int(0.04 * f0 / df))          # +-4% of f0 around harmonic
        c = int(round(fk / df))
        harm += S[max(0, c - w_h): c + w_h + 1].sum()
        mid = (k + 0.5) * f0
        cm = int(round(mid / df))
        w_i = max(2, int(0.15 * f0 / df))
        if cm + w_i < len(S):
            inter += S[cm - w_i: cm + w_i + 1].sum() * (w_h / w_i)
        k += 1
    if inter <= 0 or harm <= 0:
        return None
    return 10 * np.log10(harm / inter)


# --------------------------------------------------------------- envelope
def spectral_envelope_db(frame, sr, order=60):
    """Cepstral-liftered envelope (dB) of one frame."""
    w = frame * np.hanning(len(frame))
    N = int(2 ** np.ceil(np.log2(len(w))))
    logS = np.log(np.abs(np.fft.rfft(w, N)) + 1e-9)
    ceps = np.fft.irfft(logS)
    ceps[order:-order] = 0
    return 20 / np.log(10) * np.fft.rfft(ceps).real, sr / N


def envelope_lsd_db(x_in, y_out, sr_in, sr_out, stretch,
                    frame_s=0.046, flo=200.0, fhi=5000.0):
    """Mean log-spectral distance between envelopes on time-mapped frames."""
    n_in = int(frame_s * sr_in)
    n_out = int(frame_s * sr_out)
    hops = np.arange(0.2, (len(x_in) / sr_in) - frame_s - 0.2, 0.05)
    dists = []
    for t0 in hops:
        i = int(t0 * sr_in)
        o = int(t0 * stretch * sr_out)
        fi = x_in[i: i + n_in]
        fo = y_out[o: o + n_out]
        if len(fi) < n_in or len(fo) < n_out:
            continue
        if np.sqrt(np.mean(fi ** 2)) < 1e-4:  # skip silence
            continue
        Ei, dfi = spectral_envelope_db(fi, sr_in)
        Eo, dfo = spectral_envelope_db(fo, sr_out)
        bi = np.arange(int(flo / dfi), int(fhi / dfi))
        freqs = bi * dfi
        Eo_i = np.interp(freqs, np.arange(len(Eo)) * dfo, Eo)
        d = Ei[bi] - Eo_i
        d = d - d.mean()  # ignore broadband gain
        dists.append(np.sqrt(np.mean(d ** 2)))
    return float(np.median(dists)) if dists else None


# --------------------------------------------------------------- misc
def ltas_dist_db(x_in, y_out, sr_in, sr_out, n_bands=42):
    """1/3-octave-ish long-term spectrum distance (valid for stretch-only)."""
    def ltas(x, sr):
        f, t, Z = stft(x, sr, nperseg=4096, noverlap=2048)
        P = (np.abs(Z) ** 2).mean(axis=1)
        edges = np.geomspace(50, 16000, n_bands + 1)
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (f >= a) & (f < b)
            out.append(P[m].mean() if m.any() else np.nan)
        return 10 * np.log10(np.array(out) + EPS)
    A, B = ltas(x_in, sr_in), ltas(y_out, sr_out)
    d = A - B
    d = d[~np.isnan(d)]
    d = d - d.mean()
    return float(np.sqrt(np.mean(d ** 2)))


def warble_db(y_out, sr):
    """Spurious 2-16 Hz AM on a steady tone: modulation power vs DC."""
    env = np.abs(y_out)
    k = max(1, int(0.005 * sr))
    env = np.convolve(env, np.ones(k) / k, mode="same")[:: k]
    fs2 = sr / k
    seg = env[int(0.5 * fs2): int(0.5 * fs2) + int(min(4 * fs2, len(env) - fs2))]
    if len(seg) < 64:
        return None
    seg = seg - seg.mean()
    N = int(2 ** np.ceil(np.log2(len(seg))))
    M = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), N)) ** 2
    fm = np.fft.rfftfreq(N, 1 / fs2)
    band = (fm >= 2) & (fm <= 16)
    mod_pw = M[band].sum()
    dc_pw = (np.abs(y_out) ** 2).mean() * len(seg) / 2
    return float(10 * np.log10(mod_pw / (dc_pw + EPS) + EPS))


def stereo_coh_drift(x_in, y_out, sr_in, sr_out):
    if x_in.shape[0] < 2 or y_out.shape[0] < 2:
        return None
    def mean_coh(x, sr):
        f, C = coherence(x[0], x[1], sr, nperseg=2048)
        m = (f > 100) & (f < 8000)
        return C[m].mean()
    return float(mean_coh(x_in, sr_in) - mean_coh(y_out, sr_out))


# --------------------------------------------------------------- driver
def evaluate(in_path, out_path, stretch, pitch_semitones,
             signal_class, f0_in=None, formant_mode=False, truth_onsets=None):
    """Returns dict of applicable metrics for one rendered file."""
    x, sr_i = load(in_path)
    y, sr_o = load(out_path)
    xm, ym = mono(x), mono(y)
    res = {}
    exp_dur = (x.shape[1] / sr_i) * stretch
    res["duration_err"] = abs(y.shape[1] / sr_o - exp_dur) / exp_dur

    if signal_class in ("percussive", "mix"):
        res.update(onset_metrics(xm, ym, sr_i, sr_o, stretch, truth_onsets))

    if signal_class in ("tonal", "voice") and f0_in:
        f0_exp = f0_in * 2 ** (pitch_semitones / 12)
        f0_meas = measure_f0_fft(ym, sr_o, fmin=f0_exp * 0.7, fmax=f0_exp * 1.4)
        if f0_meas:
            res["f0_err_cents"] = 1200 * np.log2(f0_meas / f0_exp)
            h_out = hnr_db(ym, sr_o, f0_meas)
            h_in = hnr_db(xm, sr_i, f0_in)
            if h_out is not None and h_in is not None:
                res["hnr_db"] = h_out
                res["hnr_drop_db"] = h_in - h_out

    if signal_class == "voice" and (formant_mode or pitch_semitones == 0):
        res["envelope_lsd_db"] = envelope_lsd_db(xm, ym, sr_i, sr_o, stretch)

    if pitch_semitones == 0:
        res["ltas_dist_db"] = ltas_dist_db(xm, ym, sr_i, sr_o)

    if signal_class == "tonal":
        res["warble_db"] = warble_db(ym, sr_o)

    if x.shape[0] == 2:
        res["stereo_coh_drift"] = stereo_coh_drift(x, y, sr_i, sr_o)
    return res
