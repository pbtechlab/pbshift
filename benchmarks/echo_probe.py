#!/usr/bin/env python3
"""Quantify a 'fine echo / comb / reverberant-smear' artifact in a time-stretch.

The two engines under comparison (pbshift, Bungee, ...) time-stretch the SAME
voice 2.0x with pitch=0, so they are NOT sample-aligned to the input and cannot
be diff'd. We therefore measure INTRINSIC echo/comb signatures that an echo
imprints on a single signal, and compare them across engines. Because every
output is a stretch of the SAME source at the SAME pitch, the natural voice
structure (pitch period, formants) is shared; any EXTRA echo/comb structure in
one engine relative to the others (and to the input) is the artifact.

An echo  y(t) = x(t) + a*x(t-tau)  multiplies the power spectrum by
    |1 + a e^{-j w tau}|^2 = 1 + a^2 + 2a cos(w tau)
so log|Y(w)| gains a term periodic in w with period 2*pi/tau. That periodic
ripple is a COMB filter, and its real cepstrum (IFFT of log-magnitude) shows a
sharp peak at quefrency = tau. A fixed-delay echo (fixed phase-vocoder hop)
produces a peak at a CONSTANT tau, whereas the voice pitch drifts with
intonation and smears when averaged over the utterance -- so frame-averaged
cepstrum favours the fixed echo over the varying pitch.

Metrics (all reference-free, computed identically for every engine + input):
  1. Cepstral echo peak     : strongest cepstral peak in the echo band (1-30ms),
                              frame-averaged over voiced frames AND whole-signal.
                              Reports quefrency(ms) + strength. Also a
                              "non-pitch comb energy" that suppresses the pitch
                              quefrency + its harmonics to isolate echo/comb.
  2. Envelope autocorr echo : short-lag (1-30ms) normalized autocorrelation of
                              the (detrended) Hilbert envelope. A slap-back echo
                              raises autocorrelation at the echo lag.
  3. Modulation flutter     : fast (20-120Hz) vs slow (2-8Hz) envelope-modulation
                              energy. Reverberant smear raises fast modulation.
  4. Reverb ring / gap-fill : late-energy ratio after syllable offsets (echo
                              rings on) and envelope valley-fill (reverb fills
                              the silences between syllables).

usage: python benchmarks/echo_probe.py
"""
import os
import numpy as np
import soundfile as sf

EPS = 1e-12
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = [
    ("input",      "Sample/tda_0121_clm_n_f.wav"),
    ("pbshift",    "Sample/tda_0121_2x_pbshift.wav"),
    ("Bungee",     "Sample/tda_0121_2x_Bungee.wav"),
    ("Signalsmith","Sample/tda_0121_2x_Signalsmith.wav"),
    ("RubberBand", "Sample/tda_0121_2x_RubberBand.wav"),
    ("SoundTouch", "Sample/tda_0121_2x_SoundTouch.wav"),
]


# --------------------------------------------------------------------------- io
def load(path):
    x, sr = sf.read(os.path.join(ROOT, path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), sr


def interior(x, lo=0.10, hi=0.90):
    n = len(x)
    return x[int(lo * n):int(hi * n)]


def hilbert_env(x):
    """|analytic signal| via FFT Hilbert (no scipy dependency)."""
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


def parabolic(y, k):
    """Sub-sample peak location refinement around integer index k."""
    if k <= 0 or k >= len(y) - 1:
        return float(k), float(y[k])
    a, b, c = y[k - 1], y[k], y[k + 1]
    denom = (a - 2 * b + c)
    if abs(denom) < 1e-18:
        return float(k), float(b)
    d = 0.5 * (a - c) / denom
    return k + d, b - 0.25 * (a - c) * d


# ------------------------------------------------------- 1. cepstral echo peak
def real_cepstrum_from_logmag(logmag, N):
    """logmag is rfft-length; returns real cepstrum (length N)."""
    # symmetric even sequence -> irfft gives real cepstrum
    return np.fft.irfft(logmag, N)


def frame_avg_cepstrum(x, sr, N=8192, frame=4096, hop=1024, gate=0.15):
    """Average real cepstrum over voiced frames (RMS-gated)."""
    x = interior(x)
    win = np.hanning(frame)
    if len(x) < frame:
        frame = len(x) // 2 * 2
        win = np.hanning(frame)
    starts = list(range(0, len(x) - frame, hop))
    if not starts:
        return None
    rms = np.array([np.sqrt(np.mean(x[s:s + frame] ** 2)) for s in starts])
    ref = rms.max()
    acc = None
    cnt = 0
    for s, r in zip(starts, rms):
        if r < gate * ref:
            continue
        f = x[s:s + frame] * win
        spec = np.fft.rfft(f, N)
        logmag = np.log(np.abs(spec) + EPS)
        logmag = logmag - logmag.mean()          # drop DC (broadband level)
        c = real_cepstrum_from_logmag(logmag, N)
        acc = c if acc is None else acc + c
        cnt += 1
    if cnt == 0:
        return None
    return acc[:N // 2] / cnt


def whole_cepstrum(x, sr, N=None):
    x = interior(x)
    if N is None:
        N = int(2 ** np.ceil(np.log2(len(x))))
    win = np.hanning(len(x))
    spec = np.fft.rfft(x * win, N)
    logmag = np.log(np.abs(spec) + EPS)
    logmag = logmag - logmag.mean()
    c = real_cepstrum_from_logmag(logmag, N)
    return c[:N // 2]


def peak_in_band(ceps, sr, q_lo_ms, q_hi_ms, exclude=None, guard_ms=0.7):
    """Strongest peak of |ceps| in [q_lo,q_hi] ms. exclude = list of ms to mask."""
    lo = max(1, int(q_lo_ms * 1e-3 * sr))
    hi = min(len(ceps) - 2, int(q_hi_ms * 1e-3 * sr))
    seg = np.abs(ceps[lo:hi + 1]).copy()
    idx = np.arange(lo, hi + 1)
    if exclude:
        g = int(guard_ms * 1e-3 * sr)
        for em in exclude:
            ei = int(em * 1e-3 * sr)
            mask = np.abs(idx - ei) <= g
            seg[mask] = 0.0
    if len(seg) == 0 or seg.max() <= 0:
        return 0.0, 0.0
    k = int(np.argmax(seg))
    kk, val = parabolic(np.abs(ceps), lo + k)
    return kk / sr * 1e3, abs(val)


def band_rms(ceps, sr, q_lo_ms, q_hi_ms, exclude=None, guard_ms=0.7):
    lo = max(1, int(q_lo_ms * 1e-3 * sr))
    hi = min(len(ceps) - 1, int(q_hi_ms * 1e-3 * sr))
    seg = ceps[lo:hi + 1].copy()
    idx = np.arange(lo, hi + 1)
    if exclude:
        g = int(guard_ms * 1e-3 * sr)
        for em in exclude:
            ei = int(em * 1e-3 * sr)
            seg[np.abs(idx - ei) <= g] = 0.0
    return float(np.sqrt(np.mean(seg ** 2)))


# ------------------------------------------------ 2. envelope autocorr echo
def env_autocorr_echo(x, sr, lag_lo_ms=1.5, lag_hi_ms=30.0,
                      ds_hz=4000, hp_ms=40.0):
    env = hilbert_env(interior(x))
    ds = max(1, int(round(sr / ds_hz)))
    e = env[::ds]
    esr = sr / ds
    # highpass: subtract moving average (~hp_ms) to expose short-lag structure
    w = max(3, int(hp_ms * 1e-3 * esr) | 1)
    kernel = np.ones(w) / w
    trend = np.convolve(e, kernel, mode="same")
    e = e - trend
    e = e - e.mean()
    if np.sqrt(np.mean(e ** 2)) < EPS:
        return 0.0, 0.0, None, esr
    # fft-based autocorr
    n = len(e)
    nf = int(2 ** np.ceil(np.log2(2 * n)))
    E = np.fft.rfft(e, nf)
    ac = np.fft.irfft(np.abs(E) ** 2, nf)[:n]
    ac = ac / (ac[0] + EPS)
    lo = max(1, int(lag_lo_ms * 1e-3 * esr))
    hi = min(n - 2, int(lag_hi_ms * 1e-3 * esr))
    seg = ac[lo:hi + 1]
    if len(seg) == 0:
        return 0.0, 0.0, ac, esr
    k = int(np.argmax(seg))
    kk, val = parabolic(ac, lo + k)
    return kk / esr * 1e3, float(val), ac, esr


# ----------------------------------------- 3. modulation flutter (fast/slow)
def mod_flutter(x, sr, ds_hz=1000):
    """Envelope modulation energy: fast (20-120Hz) vs slow (2-8Hz) and their
    ratio. Reverberant smear raises fast modulation relative to slow."""
    env = hilbert_env(interior(x))
    ds = max(1, int(round(sr / ds_hz)))
    e = env[::ds]
    esr = sr / ds
    e = e / (np.mean(e) + EPS)
    e = e - np.mean(e)
    E = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    f = np.fft.rfftfreq(len(e), 1 / esr)

    def rms(band):
        m = (f >= band[0]) & (f <= band[1])
        return np.sqrt(np.mean(E[m] ** 2)) if m.any() else 0.0

    slow = rms((2, 8))
    fast = rms((20, 120))
    ratio_db = 20 * np.log10((fast + EPS) / (slow + EPS))
    fast_db = 20 * np.log10(fast + 1e-9)
    return fast_db, ratio_db


# --------------------------------------- 4. reverb ring / gap-fill
def reverb_ring(x, sr):
    """Late-energy ring after syllable offsets + envelope valley-fill.

    ring_db  : median over syllable offsets of energy 20-60ms AFTER a falling
               edge relative to energy just BEFORE it -- an echo/reverb tail
               keeps energy alive after the source stops (higher = more ring).
    gapfill  : envelope 10th/90th percentile ratio (dB). Reverb/echo fills the
               inter-syllable valleys, RAISING the floor (less negative = more
               fill / smear).
    """
    env = hilbert_env(interior(x))
    ds = max(1, int(round(sr / 1000)))
    e = env[::ds]
    esr = sr / ds
    # smooth envelope a touch
    w = max(3, int(0.005 * esr) | 1)
    e = np.convolve(e, np.ones(w) / w, mode="same")
    e = e / (e.max() + EPS)

    # gap-fill: dynamic range of the (log) envelope over active region
    active = e[e > 0.02 * e.max()]
    p10 = np.percentile(active, 10)
    p90 = np.percentile(active, 90)
    gapfill_db = 20 * np.log10((p10 + EPS) / (p90 + EPS))  # closer to 0 = filled

    # ring: find falling edges (offsets) where envelope drops through 0.3*peak
    thr = 0.30
    above = e > thr
    offsets = np.where(above[:-1] & ~above[1:])[0]
    d1 = int(0.005 * esr)     # "before" window end (just before drop)
    la, lb = int(0.020 * esr), int(0.060 * esr)   # late window 20-60ms after
    rings = []
    for o in offsets:
        pre = e[max(0, o - 3 * d1):o]
        late = e[o + la:o + lb]
        if len(pre) < 2 or len(late) < 2:
            continue
        pe = np.sqrt(np.mean(pre ** 2))
        le = np.sqrt(np.mean(late ** 2))
        if pe < 1e-3:
            continue
        rings.append(20 * np.log10((le + EPS) / (pe + EPS)))
    ring_db = float(np.median(rings)) if rings else float("nan")
    return ring_db, gapfill_db, len(rings)


# --------------------------------------------------------------------- driver
def analyze():
    sr = None
    data = {}
    for name, path in FILES:
        x, s = load(path)
        sr = s
        data[name] = x

    # pitch quefrency from the INPUT (shared by all outputs; pitch preserved)
    ci = frame_avg_cepstrum(data["input"], sr)
    pitch_ms, pitch_val = peak_in_band(ci, sr, 3.0, 11.0)
    # harmonics of the pitch quefrency to exclude when isolating echo/comb
    pitch_excl = [pitch_ms, 2 * pitch_ms, 0.5 * pitch_ms]

    rows = {}
    for name, _ in FILES:
        x = data[name]
        cav = frame_avg_cepstrum(x, sr)
        cwh = whole_cepstrum(x, sr)
        # strongest cepstral peak in echo band (frame-avg), unrestricted
        ep_q, ep_v = peak_in_band(cav, sr, 1.0, 30.0)
        # strongest echo-band peak EXCLUDING pitch + harmonics (isolate comb)
        en_q, en_v = peak_in_band(cav, sr, 1.0, 30.0,
                                  exclude=pitch_excl, guard_ms=0.8)
        # non-pitch comb energy (rms of cepstrum in band w/ pitch masked)
        comb_e = band_rms(cav, sr, 1.0, 30.0, exclude=pitch_excl, guard_ms=0.8)
        # whole-signal cepstrum echo peak
        wp_q, wp_v = peak_in_band(cwh, sr, 1.0, 30.0)
        # envelope autocorr echo
        ac_lag, ac_val, _, _ = env_autocorr_echo(x, sr)
        # modulation flutter
        fast_db, ratio_db = mod_flutter(x, sr)
        # reverb ring / gapfill
        ring_db, gapfill_db, noff = reverb_ring(x, sr)

        # hop-synchronous comb readout: cepstral value at 512-sample multiples
        # (512/44100 = 11.61 ms). The multires engine hops at 512 samples, so a
        # hop-synchronous comb rings the cepstrum here + at its harmonics.
        def cval(cav, samp):
            return float(np.abs(cav[samp])) if samp < len(cav) else float("nan")
        hop_ms = 512 / sr * 1e3
        hop1 = cval(cav, 512)
        hop2 = cval(cav, 1024)

        rows[name] = dict(
            ep_q=ep_q, ep_v=ep_v, en_q=en_q, en_v=en_v, comb_e=comb_e,
            wp_q=wp_q, wp_v=wp_v, ac_lag=ac_lag, ac_val=ac_val,
            fast_db=fast_db, ratio_db=ratio_db, ring_db=ring_db,
            gapfill_db=gapfill_db, noff=noff, hop_ms=hop_ms,
            hop1=hop1, hop2=hop2,
        )
    return sr, pitch_ms, pitch_val, rows


def main():
    sr, pitch_ms, pitch_val, rows = analyze()
    order = [n for n, _ in FILES]

    print("=" * 100)
    print("ECHO / COMB / REVERB-SMEAR PROBE  (2.0x voice stretch, pitch 0)")
    print(f"sr={sr}Hz   voice pitch quefrency (from input) = {pitch_ms:.2f} ms "
          f"(f0~{1000/pitch_ms:.0f}Hz)  cepstral pitch peak={pitch_val:.3f}")
    print("=" * 100)

    def col(v, fmt):
        return fmt.format(v)

    # Table 1: cepstral echo peaks
    hdr = f"{'engine':<12}"
    for n in order:
        pass
    print("\n[1] CEPSTRAL ECHO PEAK  (real cepstrum; peak = comb at quefrency=tau)")
    print(f"{'engine':<12} {'frameAvg peak':>16} {'non-pitch comb':>18} "
          f"{'combEnergy':>11} {'whole-sig peak':>16}")
    for n in order:
        r = rows[n]
        print(f"{n:<12} "
              f"{r['ep_v']:.3f} @ {r['ep_q']:5.2f}ms "
              f"{r['en_v']:.3f} @ {r['en_q']:5.2f}ms "
              f"{r['comb_e']:>11.4f} "
              f"{r['wp_v']:.3f} @ {r['wp_q']:5.2f}ms")

    hop_ms = rows[order[0]]["hop_ms"]
    inp_hop1 = rows["input"]["hop1"]
    print(f"\n[1b] HOP-SYNCHRONOUS COMB  (cepstral value at 512-sample multiples; "
          f"hop={hop_ms:.2f}ms)")
    print(f"{'engine':<12} {'@11.6ms(512)':>13} {'@23.2ms(1024)':>14} "
          f"{'x vs input':>11}")
    for n in order:
        r = rows[n]
        print(f"{n:<12} {r['hop1']:>13.4f} {r['hop2']:>14.4f} "
              f"{r['hop1']/(inp_hop1+EPS):>10.1f}x")

    print("\n[2] ENVELOPE AUTOCORRELATION ECHO  (detrended Hilbert env, lag 1.5-30ms)")
    print(f"{'engine':<12} {'peak autocorr @ lag':>24}")
    for n in order:
        r = rows[n]
        print(f"{n:<12} {r['ac_val']:.3f} @ {r['ac_lag']:5.2f}ms")

    print("\n[3] MODULATION FLUTTER  (fast=20-120Hz smear)")
    print(f"{'engine':<12} {'fast mod(dB)':>13} {'fast/slow(dB)':>15}")
    for n in order:
        r = rows[n]
        print(f"{n:<12} {r['fast_db']:>13.1f} {r['ratio_db']:>15.1f}")

    print("\n[4] REVERB RING / GAP-FILL")
    print(f"{'engine':<12} {'ring after offset(dB)':>22} {'valley-fill(dB)':>17} "
          f"{'#offsets':>9}")
    for n in order:
        r = rows[n]
        print(f"{n:<12} {r['ring_db']:>22.1f} {r['gapfill_db']:>17.1f} "
              f"{r['noff']:>9d}")

    # ------- differential vs Bungee (the reference "clean" competitor) -------
    print("\n" + "=" * 100)
    print("DIFFERENTIAL  (pbshift MINUS Bungee; positive = pbshift more echo-like)")
    print("=" * 100)
    b = rows["Bungee"]
    p = rows["pbshift"]
    print(f"  comb energy (non-pitch)     : pbshift {p['comb_e']:.4f}  "
          f"Bungee {b['comb_e']:.4f}   delta {p['comb_e']-b['comb_e']:+.4f}  "
          f"ratio x{p['comb_e']/(b['comb_e']+EPS):.2f}")
    print(f"  non-pitch comb peak         : pbshift {p['en_v']:.3f}@{p['en_q']:.2f}ms  "
          f"Bungee {b['en_v']:.3f}@{b['en_q']:.2f}ms")
    print(f"  env autocorr echo           : pbshift {p['ac_val']:.3f}@{p['ac_lag']:.2f}ms  "
          f"Bungee {b['ac_val']:.3f}@{b['ac_lag']:.2f}ms   "
          f"delta {p['ac_val']-b['ac_val']:+.3f}")
    print(f"  fast/slow modulation        : pbshift {p['ratio_db']:+.1f}dB  "
          f"Bungee {b['ratio_db']:+.1f}dB   delta {p['ratio_db']-b['ratio_db']:+.1f}dB")
    print(f"  valley-fill (gap smear)     : pbshift {p['gapfill_db']:+.1f}dB  "
          f"Bungee {b['gapfill_db']:+.1f}dB   delta {p['gapfill_db']-b['gapfill_db']:+.1f}dB")

    return sr, pitch_ms, rows


if __name__ == "__main__":
    main()
