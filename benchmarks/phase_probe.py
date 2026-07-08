#!/usr/bin/env python3
"""PHASE / TEMPORAL probe for time-stretch artifacts a magnitude cepstrum misses.

Context
-------
A human A/B test on 2.0x VOICE time-stretch ranks Bungee clearly best and hears
pbshift (both the old default and the phase-"coherent" v2) as having more
"delay-feel" and worse quality. Yet every MAGNITUDE-domain metric we built
(echo_probe.py: cepstral comb, envelope autocorr, modulation flutter, reverb
ring) now says pb_v2 >= Bungee. The perceptual gap therefore lives in a
dimension a magnitude cepstrum is BLIND to: PHASE dispersion / temporal smear /
transient sharpness. A long synthesis window phase-vocoder (pbshift: symmetric
N=4096 analysis+synthesis) spreads each glottal pulse and each attack over the
window, producing a "watery / reverberant / delayed" excitation even when the
short-time MAGNITUDE spectrum (and hence the cepstrum) is perfect. Bungee uses a
SHORT asymmetric OUTPUT window (2048) -> sharper, more concentrated excitation.

All four signals are 2.0x stretches of the SAME voice at the SAME pitch, so they
are NOT sample-aligned and cannot be diff'd. We therefore measure INTRINSIC,
reference-free, phase/temporal signatures, computed IDENTICALLY for every file,
and read the ORDERING. The input (un-stretched) is the gold standard: whichever
output's number is CLOSER to the input is more faithful.

Every metric below is designed to be (largely) MAGNITUDE-INVARIANT so it cannot
be "gamed" by the magnitude-domain fixes that already saturated:
  - LPC/cepstral-liftering removes the vocal-tract MAGNITUDE envelope, leaving an
    excitation whose PEAKINESS is a pure phase/temporal property.
  - crest factor, kurtosis, group delay, harmonic phase-curvature are all
    invariant to overall level and (for the phase ones) to the magnitude spectrum.

Metric families
---------------
 1. Excitation peakiness   : crest factor & kurtosis of the LPC residual and of
                             the cepstrum-liftered excitation over voiced frames.
                             A coherent glottal pulse train is impulsive (high
                             crest / high kurtosis); phase dispersion smears it.
 2. Pitch-sync crest       : peak-to-RMS of the raw waveform inside one-pitch-
                             period windows over voiced frames. Dispersion spreads
                             the glottal pulse across the period -> lower crest.
 3. Group-delay dispersion : magnitude-weighted std of the short-time group delay
                             across frequency within voiced frames (ms). A single
                             glottal instant => all frequencies share one delay
                             (low spread); a "reverberant" PV smears delay across
                             frequency (high spread).
 4. Harmonic phase curvature: circular std of the 2nd difference of harmonic phase
                             (removes constant + linear-timing terms, isolating
                             vertical phase dispersion = Laroche-Dolson phasiness).
 5. Transient/onset + pre-echo: 10-90% attack rise-time, attack log-slope, and
                             pre-echo (energy leaking BEFORE the true attack). A
                             long PV window smears/pre-rings attacks.
 6. Intra-phoneme EDT      : Schroeder early-decay time after voiced offsets and
                             onset->offset asymmetry. Smear/echo lengthens decay.

usage: python benchmarks/phase_probe.py
"""
import os
import numpy as np
import soundfile as sf
from scipy.signal import lfilter
from scipy.stats import kurtosis as _kurtosis

EPS = 1e-12
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = [
    ("input",      "Sample/tda_0121_clm_n_f.wav"),   # gold standard (un-stretched)
    ("pb_default", "Sample/tda_0121_2x_pbshift.wav"),      # old default (mag comb)
    ("pb_v2",      "Sample/tda_0121_2x_pbshift_v2.wav"),   # PBSHIFT_COHERENT=1
    ("Bungee",     "Sample/tda_0121_2x_Bungee.wav"),       # human's favourite
]


# --------------------------------------------------------------------------- io
def load(path):
    x, sr = sf.read(os.path.join(ROOT, path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), sr


def interior(x, lo=0.05, hi=0.95):
    n = len(x)
    return x[int(lo * n):int(hi * n)]


def hilbert_env(x):
    n = len(x)
    X = np.fft.fft(x)
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(X * h))


# ---------------------------------------------------------- voiced-frame helper
def voiced_frames(x, sr, frame, hop, gate=0.15, f0_lo=80, f0_hi=400,
                  ac_thresh=0.30):
    """Yield (start, frame_samples, T0_samples) for RMS-gated, clearly-voiced
    frames. T0 from autocorrelation peak in [f0_hi,f0_lo]."""
    x = interior(x)
    win = np.hanning(frame)
    starts = list(range(0, len(x) - frame, hop))
    rms = np.array([np.sqrt(np.mean(x[s:s + frame] ** 2)) for s in starts])
    ref = rms.max()
    lo = int(sr / f0_hi)
    hi = int(sr / f0_lo)
    out = []
    for s, r in zip(starts, rms):
        if r < gate * ref:
            continue
        f = x[s:s + frame]
        fw = f * win
        ac = np.correlate(fw, fw, "full")[frame - 1:]
        seg = ac[lo:hi]
        if len(seg) == 0:
            continue
        k = lo + int(np.argmax(seg))
        if ac[0] <= 0 or ac[k] < ac_thresh * ac[0]:
            continue                       # not clearly periodic -> skip
        out.append((s, f, k))
    return out


# ----------------------------------------- 1. excitation peakiness (LPC / lifter)
def _lpc(frame, order):
    """Levinson-Durbin LPC from autocorrelation. Returns a (len order+1)."""
    r = np.correlate(frame, frame, "full")[len(frame) - 1:len(frame) + order]
    if r[0] <= 0:
        return None
    r = r + np.concatenate(([r[0] * 1e-6], np.zeros(order)))  # tiny ridge
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1])
        k = -acc / e
        a[1:i + 1] = a[1:i + 1] + k * a[i - 1::-1][:i]
        e *= (1 - k * k)
        if e <= 0:
            break
    return a


def excitation_peakiness(x, sr, frame=1536, hop=512, order=24):
    """Crest & kurtosis of the LPC residual over voiced frames. LPC removes the
    MAGNITUDE spectral envelope, so residual peakiness is a phase/temporal-only
    property: an impulsive glottal train => high crest/kurtosis; dispersion
    (watery PV) => low. Also a cepstrum-liftered excitation crest as a check."""
    frs = voiced_frames(x, sr, frame, hop)
    if not frs:
        return dict(res_crest=np.nan, res_kurt=np.nan, lift_crest=np.nan, n=0)
    win = np.hanning(frame)
    res_crest, res_kurt, lift_crest = [], [], []
    for s, f, T0 in frs:
        fw = f * win
        a = _lpc(fw, order)
        if a is None:
            continue
        e = lfilter(a, [1.0], fw)
        core = e[order:frame - order]           # drop filter transient edges
        if len(core) < T0 * 2:
            continue
        rms = np.sqrt(np.mean(core ** 2))
        if rms < EPS:
            continue
        res_crest.append(np.max(np.abs(core)) / rms)
        res_kurt.append(_kurtosis(core, fisher=True))
        # cepstral-lifter excitation: zero low quefrencies (envelope) -> excitation
        N = 4096
        spec = np.fft.rfft(fw, N)
        logm = np.log(np.abs(spec) + EPS)
        cep = np.fft.irfft(logm, N)
        lift = np.ones(N)
        Lq = int(0.002 * sr)                    # 2 ms lifter: below = envelope
        lift[:Lq] = 0.0
        lift[-Lq + 1:] = 0.0
        cep_h = cep * lift
        exc_logm = np.fft.rfft(cep_h, N)[:len(spec)].real
        exc = np.fft.irfft(np.exp(1j * np.angle(spec)) * np.exp(exc_logm), N)
        exc = exc[order:frame - order]
        er = np.sqrt(np.mean(exc ** 2))
        if er > EPS:
            lift_crest.append(np.max(np.abs(exc)) / er)
    return dict(
        res_crest=float(np.mean(res_crest)) if res_crest else np.nan,
        res_kurt=float(np.mean(res_kurt)) if res_kurt else np.nan,
        lift_crest=float(np.mean(lift_crest)) if lift_crest else np.nan,
        n=len(res_crest),
    )


# --------------------------------------------- 2. pitch-synchronous crest factor
def pitchsync_crest(x, sr, frame=1536, hop=512):
    """Peak-to-RMS of the RAW waveform in one-pitch-period windows over voiced
    frames, and the fraction of period energy in its top 20%% samples. A coherent
    glottal pulse => energy concentrated in a small part of the period (high
    crest, high concentration); dispersion spreads it (low)."""
    frs = voiced_frames(x, sr, frame, hop)
    if not frs:
        return dict(ps_crest=np.nan, ps_conc=np.nan, n=0)
    crests, concs = [], []
    for s, f, T0 in frs:
        # slide one-period windows across the frame, average their crest
        for p in range(0, frame - T0, T0):
            w = f[p:p + T0]
            rms = np.sqrt(np.mean(w ** 2))
            if rms < EPS:
                continue
            crests.append(np.max(np.abs(w)) / rms)
            # energy concentration: fraction of total energy in top 20% samples
            e = np.sort(w ** 2)[::-1]
            k = max(1, int(0.20 * len(e)))
            concs.append(e[:k].sum() / (e.sum() + EPS))
    return dict(
        ps_crest=float(np.mean(crests)) if crests else np.nan,
        ps_conc=float(np.mean(concs)) if concs else np.nan,
        n=len(crests),
    )


# ------------------------------------------------- 3. group-delay dispersion
def groupdelay_dispersion(x, sr, frame=2048, hop=512, band=(150, 5000)):
    """Magnitude-weighted std of short-time group delay across frequency within
    voiced frames (ms). tau_g(w) = Re(FFT(n*x)*conj(FFT(x)))/|FFT(x)|^2. A single
    glottal instant => nearly constant delay across frequency (low spread); a
    dispersive/reverberant PV smears the delay across frequency (high spread)."""
    frs = voiced_frames(x, sr, frame, hop)
    if not frs:
        return dict(gd_std=np.nan, gd_iqr=np.nan, n=0)
    win = np.hanning(frame)
    n_idx = np.arange(frame)
    freqs = np.fft.rfftfreq(frame, 1 / sr)
    bmask = (freqs >= band[0]) & (freqs <= band[1])
    stds, iqrs = [], []
    for s, f, T0 in frs:
        fw = f * win
        X = np.fft.rfft(fw)
        Xn = np.fft.rfft(n_idx * fw)
        mag2 = (X.real ** 2 + X.imag ** 2)
        tau = np.real(Xn * np.conj(X)) / (mag2 + EPS)      # samples
        tau_ms = tau / sr * 1e3
        w = mag2[bmask]
        t = tau_ms[bmask]
        w = w / (w.sum() + EPS)
        mean = np.sum(w * t)
        var = np.sum(w * (t - mean) ** 2)
        stds.append(np.sqrt(var))
        # weighted IQR proxy: spread of the top-magnitude (harmonic) bins only
        order = np.argsort(mag2[bmask])[::-1]
        top = t[order[:max(4, len(order) // 8)]]
        iqrs.append(np.percentile(top, 75) - np.percentile(top, 25))
    return dict(
        gd_std=float(np.mean(stds)) if stds else np.nan,
        gd_iqr=float(np.mean(iqrs)) if iqrs else np.nan,
        n=len(stds),
    )


# ------------------------------------------- 4. harmonic phase curvature (phasiness)
def _circ_std(angles):
    R = np.abs(np.mean(np.exp(1j * angles)))
    R = min(max(R, EPS), 1.0)
    return np.sqrt(-2.0 * np.log(R))            # radians


def harmonic_phase_curvature(x, sr, frame=2048, hop=512, nharm=20,
                             f0_lo=80, f0_hi=400):
    """Circular std of the 2nd difference of harmonic phase across a voiced frame,
    averaged over frames. Delta2_h = phi_{h+1} - 2 phi_h + phi_{h-1} removes the
    constant (irrelevant) AND linear (glottal-instant timing) phase terms,
    leaving only the CURVATURE = vertical phase dispersion (Laroche-Dolson
    'phasiness'). A clean periodic pulse has smooth harmonic phase (low curvature
    spread); a phasey PV randomizes it (high)."""
    frs = voiced_frames(x, sr, frame, hop)
    if not frs:
        return dict(phi_curv=np.nan, n=0)
    win = np.hanning(frame)
    curvs = []
    for s, f, T0 in frs:
        f0 = sr / T0
        fw = f * win
        X = np.fft.rfft(fw)
        mag = np.abs(X)
        ph = np.angle(X)
        freqs = np.fft.rfftfreq(frame, 1 / sr)
        # locate harmonics: nearest bin to h*f0, refine to local mag peak +-2 bins
        hp = []
        for h in range(1, nharm + 1):
            fh = h * f0
            if fh > 5000 or fh > sr / 2:
                break
            k0 = int(round(fh * frame / sr))
            k = max(1, min(len(mag) - 2, k0))
            lo = max(1, k - 2)
            hi = min(len(mag) - 1, k + 3)
            k = lo + int(np.argmax(mag[lo:hi]))
            hp.append(ph[k])
        if len(hp) < 4:
            continue
        hp = np.unwrap(np.array(hp))
        d2 = hp[2:] - 2 * hp[1:-1] + hp[:-2]        # 2nd difference
        # wrap to [-pi,pi] and take circular spread
        d2 = np.angle(np.exp(1j * d2))
        curvs.append(_circ_std(d2))
    return dict(phi_curv=float(np.mean(curvs)) if curvs else np.nan,
                n=len(curvs))


# ------------------------------------------- 5. transient sharpness + pre-echo
def transient_preecho(x, sr):
    """Onset attack sharpness and pre-echo. Envelope onsets are detected on a 1kHz
    log-envelope; for each we measure 10-90%% rise time (ms), the max log-envelope
    slope (attack steepness), and a PRE-ECHO ratio = energy 5-20ms BEFORE the
    attack relative to energy 30-60ms before (a quiet region). A long-window PV
    leaks energy before the attack (pre-echo) and lengthens the rise."""
    env = hilbert_env(interior(x))
    ds = max(1, int(round(sr / 2000)))          # 2 kHz envelope
    e = env[::ds]
    esr = sr / ds
    w = max(3, int(0.003 * esr) | 1)
    e = np.convolve(e, np.ones(w) / w, mode="same")
    e = e / (e.max() + EPS)
    le = np.log(e + 1e-4)
    # onsets: rising through 0.25 of peak with a low pre-level
    thr = 0.25
    above = e > thr
    onsets = np.where(~above[:-1] & above[1:])[0]
    rises, slopes, preechos = [], [], []
    for o in onsets:
        # find local peak within 40ms after onset
        pa = e[o:o + int(0.040 * esr)]
        if len(pa) < 3:
            continue
        pk = e[o + int(np.argmax(pa))]
        if pk < 0.20:
            continue
        # rise time 10->90% searching backward/forward around o
        seg_lo = max(0, o - int(0.030 * esr))
        seg_hi = min(len(e), o + int(0.040 * esr))
        seg = e[seg_lo:seg_hi]
        lo_thr, hi_thr = 0.10 * pk, 0.90 * pk
        idx_hi = np.where(seg >= hi_thr)[0]
        idx_lo = np.where(seg >= lo_thr)[0]
        if len(idx_hi) == 0 or len(idx_lo) == 0:
            continue
        t_hi = idx_hi[0]
        t_lo = idx_lo[0]
        if t_hi <= t_lo:
            continue
        rises.append((t_hi - t_lo) / esr * 1e3)
        # attack slope: max positive diff of log-envelope in the rise
        d = np.diff(le[seg_lo:seg_hi]) * esr
        slopes.append(np.max(d))
        # pre-echo: energy just before attack vs quiet baseline further back
        near = e[max(0, o - int(0.020 * esr)):max(1, o - int(0.005 * esr))]
        far = e[max(0, o - int(0.060 * esr)):max(1, o - int(0.030 * esr))]
        if len(near) > 1 and len(far) > 1:
            nr = np.sqrt(np.mean(near ** 2))
            fr = np.sqrt(np.mean(far ** 2))
            preechos.append(20 * np.log10((nr + EPS) / (fr + EPS)))
    return dict(
        rise_ms=float(np.median(rises)) if rises else np.nan,
        atk_slope=float(np.median(slopes)) if slopes else np.nan,
        preecho_db=float(np.median(preechos)) if preechos else np.nan,
        n=len(rises),
    )


# ------------------------------------------------ 6. intra-phoneme EDT / decay
def intra_edt(x, sr):
    """Schroeder early-decay after voiced offsets. For each falling edge we
    backward-integrate the local envelope energy and fit the 0 -> -10 dB decay;
    the extrapolated 60 dB time (EDT) is longer when smear/echo rings on. Also
    reports onset/offset slope asymmetry (a smeared engine softens both, but the
    decay side is where reverberant tails live)."""
    env = hilbert_env(interior(x))
    ds = max(1, int(round(sr / 2000)))
    e = env[::ds]
    esr = sr / ds
    w = max(3, int(0.003 * esr) | 1)
    e = np.convolve(e, np.ones(w) / w, mode="same")
    e = e / (e.max() + EPS)
    thr = 0.30
    above = e > thr
    offsets = np.where(above[:-1] & ~above[1:])[0]
    edts = []
    for o in offsets:
        seg = e[o:o + int(0.120 * esr)]         # 120 ms decay window
        if len(seg) < int(0.030 * esr):
            continue
        # Schroeder backward energy integral
        p = seg ** 2
        sch = np.cumsum(p[::-1])[::-1]
        sch = 10 * np.log10(sch / (sch[0] + EPS) + EPS)
        # fit 0 -> -10 dB region -> EDT (x6 for 60 dB)
        i10 = np.where(sch <= -10)[0]
        if len(i10) == 0:
            continue
        t10 = i10[0] / esr
        edts.append(t10 * 6.0 * 1e3)            # EDT in ms
    return dict(edt_ms=float(np.median(edts)) if edts else np.nan,
                n=len(edts))


# --------------------------------------------------------------------- driver
def analyze():
    sr = None
    rows = {}
    for name, path in FILES:
        x, s = load(path)
        sr = s
        r = {}
        r.update(excitation_peakiness(x, sr))
        r.update(pitchsync_crest(x, sr))
        r.update(groupdelay_dispersion(x, sr))
        r.update(harmonic_phase_curvature(x, sr))
        r.update(transient_preecho(x, sr))
        r.update(intra_edt(x, sr))
        rows[name] = r
    return sr, rows


def _fmt(v, f="{:.3f}"):
    return "  n/a " if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def main():
    sr, rows = analyze()
    order = [n for n, _ in FILES]
    inp = rows["input"]

    print("=" * 104)
    print("PHASE / TEMPORAL PROBE  (2.0x voice stretch, pitch 0)   sr=%dHz" % sr)
    print("input = un-stretched gold standard; CLOSER-to-input is more faithful")
    print("Human ordering to reproduce:  Bungee  >  pb_v2  >=  pb_default")
    print("=" * 104)

    def table(title, cols):
        print("\n" + title)
        head = f"{'engine':<12}" + "".join(f"{c[0]:>16}" for c in cols)
        print(head)
        for n in order:
            r = rows[n]
            line = f"{n:<12}"
            for _, key, fmt in cols:
                line += f"{_fmt(r.get(key), fmt):>16}"
            print(line)

    table("[1] EXCITATION PEAKINESS  (LPC removes magnitude envelope; HIGHER = more"
          " impulsive/coherent)",
          [("res crest", "res_crest", "{:.3f}"),
           ("res kurtosis", "res_kurt", "{:.2f}"),
           ("lifter crest", "lift_crest", "{:.3f}"),
           ("nframes", "n", "{:.0f}")])

    table("[2] PITCH-SYNC CREST  (raw waveform per pitch period; HIGHER = pulse more"
          " concentrated)",
          [("ps crest", "ps_crest", "{:.3f}"),
           ("top20% energy", "ps_conc", "{:.3f}")])

    table("[3] GROUP-DELAY DISPERSION  (mag-weighted, ms; LOWER = phase aligned"
          " across freq)",
          [("gd std (ms)", "gd_std", "{:.3f}"),
           ("gd IQR (ms)", "gd_iqr", "{:.3f}")])

    table("[4] HARMONIC PHASE CURVATURE  (circ std of 2nd diff, rad; LOWER = less"
          " phasiness)",
          [("phi curv (rad)", "phi_curv", "{:.4f}")])

    table("[5] TRANSIENT / PRE-ECHO  (rise LOWER=sharper; slope HIGHER=sharper;"
          " preecho LOWER=cleaner)",
          [("rise 10-90(ms)", "rise_ms", "{:.2f}"),
           ("atk slope", "atk_slope", "{:.1f}"),
           ("pre-echo (dB)", "preecho_db", "{:.2f}")])

    table("[6] INTRA-PHONEME EDT  (early decay time, ms; LOWER = less reverberant"
          " ring)",
          [("EDT (ms)", "edt_ms", "{:.1f}")])

    # ---- perception verdict: which metrics reproduce Bungee > pb_v2 -----------
    print("\n" + "=" * 104)
    print("PERCEPTION TRACKING  (does the metric rank Bungee BETTER than pb_v2?)")
    print("  a metric TRACKS perception iff Bungee is more faithful than pb_v2")
    print("=" * 104)

    # eval mode = how "cleaner/more faithful" is defined for each metric:
    #   'high' = higher is cleaner ; 'low' = lower is cleaner
    #   'inp'  = closeness to the un-stretched INPUT value is cleaner (the input
    #            is the gold standard; for some phase measures the natural voice
    #            is NOT at an extreme, so an extreme is itself the artifact).
    checks = [
        ("res_crest",  "high", "excitation LPC-residual crest"),
        ("res_kurt",   "high", "excitation LPC-residual kurtosis"),
        ("lift_crest", "high", "cepstrum-lifter excitation crest"),
        ("ps_crest",   "inp",  "pitch-sync waveform crest"),
        ("ps_conc",    "inp",  "pitch-period top-20% energy conc."),
        ("gd_std",     "inp",  "group-delay dispersion (ms)"),
        ("gd_iqr",     "inp",  "group-delay IQR (ms)"),
        ("phi_curv",   "inp",  "harmonic phase curvature (rad)"),
        ("rise_ms",    "low",  "attack rise time (ms)"),
        ("atk_slope",  "high", "attack log-slope"),
        ("preecho_db", "low",  "pre-echo (dB)"),
        ("edt_ms",     "inp",  "intra-phoneme EDT (ms)"),
    ]
    b = rows["Bungee"]
    v = rows["pb_v2"]
    d = rows["pb_default"]

    def score(mode, val, iv):
        """Lower score = cleaner/more faithful (so we can rank engines)."""
        if mode == "high":
            return -val
        if mode == "low":
            return val
        return abs(val - iv)          # 'inp': deviation from input

    print(f"{'metric':<38}{'pb_default':>11}{'pb_v2':>11}{'Bungee':>11}"
          f"{'input':>11}  {'engine order (best>worst)':>26}  {'B>v2?':>6}")
    tracking = []
    for key, mode, label in checks:
        bv, vv, dv, iv = b.get(key), v.get(key), d.get(key), inp.get(key)
        if any(x is None or (isinstance(x, float) and np.isnan(x))
               for x in (bv, vv, dv)):
            print(f"{label:<38}{_fmt(dv):>11}{_fmt(vv):>11}{_fmt(bv):>11}"
                  f"{_fmt(iv):>11}  {'n/a':>26}  {'n/a':>6}")
            continue
        eng = {"pb_default": score(mode, dv, iv),
               "pb_v2": score(mode, vv, iv),
               "Bungee": score(mode, bv, iv)}
        ordered = sorted(eng, key=lambda k: eng[k])    # best first
        b_better = eng["Bungee"] < eng["pb_v2"]
        # deviation-from-input magnitudes (the "artifact" size) when mode='inp'
        if mode == "inp":
            dev = {k: abs({'pb_default':dv,'pb_v2':vv,'Bungee':bv}[k] - iv)
                   for k in eng}
            margin = abs(dev["pb_v2"] - dev["Bungee"]) / (dev["pb_v2"] + EPS) * 100
        else:
            margin = abs(bv - vv) / (abs(vv) + EPS) * 100
        matches_full = ordered == ["Bungee", "pb_v2", "pb_default"]
        tag = ("YES" if b_better else "no")
        print(f"{label:<38}{_fmt(dv):>11}{_fmt(vv):>11}{_fmt(bv):>11}"
              f"{_fmt(iv):>11}  {'>'.join(ordered):>26}  {tag:>6}")
        if b_better:
            tracking.append((label, mode, dv, vv, bv, iv, margin,
                             matches_full))

    print("\n---- METRICS THAT TRACK PERCEPTION (Bungee more faithful than pb_v2) ----")
    print("  ** = also reproduces the FULL human order Bungee > pb_v2 > pb_default")
    if not tracking:
        print("  NONE - every phase metric still ranks pb_v2 >= Bungee.")
    for label, mode, dv, vv, bv, iv, margin, full in sorted(
            tracking, key=lambda t: (-int(t[-1]), -t[-2])):
        star = " **" if full else "   "
        if mode == "inp":
            print(f" {star}{label:<38} dev-from-input: Bungee {abs(bv-iv):.3f} "
                  f"< pb_v2 {abs(vv-iv):.3f} < pb_default {abs(dv-iv):.3f} ms/units"
                  f"  ({margin:.0f}% closer); raw B={bv:.3f} v2={vv:.3f} inp={iv:.3f}")
        else:
            arrow = "higher" if mode == "high" else "lower"
            print(f" {star}{label:<38} Bungee {bv:.3f} vs pb_v2 {vv:.3f} "
                  f"({arrow}=cleaner, {margin:.0f}% margin); pb_default {_fmt(dv)}")

    print("\n---- METRICS THAT FAIL (rank pb_v2 >= Bungee -> same trap as the cepstrum) ----")
    for key, mode, label in checks:
        bv, vv, dv, iv = b.get(key), v.get(key), d.get(key), inp.get(key)
        if any(x is None or (isinstance(x, float) and np.isnan(x))
               for x in (bv, vv, dv)):
            continue
        eng = {"pb_default": score(mode, dv, iv), "pb_v2": score(mode, vv, iv),
               "Bungee": score(mode, bv, iv)}
        if not (eng["Bungee"] < eng["pb_v2"]):
            print(f"  {label:<38} pb_v2 {vv:.3f} ranked >= Bungee {bv:.3f}")

    return sr, rows


if __name__ == "__main__":
    main()
