# -*- coding: utf-8 -*-
"""
Test-signal corpus generator for pbPitchShift benchmarking.
Deterministic (fixed seed). 48 kHz float32 WAV.

Signal classes follow TSM-literature practice (Driedger/Mueller TSM toolbox):
solo voice / percussive / polyphonic / plucked / pad + analytic tones.
"""
import json
import numpy as np
import soundfile as sf
import subprocess
import sys
from pathlib import Path

SR = 48000
OUT = Path(__file__).parent / "corpus"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260705)
TRUTH = {}  # signal -> {"onsets": [seconds]}


def save(name, x, sr=SR):
    x = np.asarray(x, dtype=np.float32)
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x * (0.708 / peak)  # -3 dBFS
    sf.write(OUT / f"{name}.wav", x.T if x.ndim == 2 else x, sr, subtype="FLOAT")
    dur = (x.shape[-1] if x.ndim == 2 else len(x)) / sr
    print(f"  {name}.wav  {dur:.2f}s  {'stereo' if x.ndim == 2 else 'mono'}")


def adsr(n, a, d, s_level, r, sr=SR):
    a_n, d_n, r_n = int(a * sr), int(d * sr), int(r * sr)
    s_n = max(0, n - a_n - d_n - r_n)
    env = np.concatenate([
        np.linspace(0, 1, max(a_n, 1)),
        np.linspace(1, s_level, max(d_n, 1)),
        np.full(s_n, s_level),
        np.linspace(s_level, 0, max(r_n, 1)),
    ])
    return env[:n] if len(env) >= n else np.pad(env, (0, n - len(env)))


# ---------------------------------------------------------------- voice (TTS)
def gen_tts_speech():
    """Real speech via Windows SAPI (reproducible, license-free)."""
    ps = r'''
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = 0
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(48000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile("{out}", $fmt)
$s.Speak("{text}")
$s.Dispose()
'''
    tmp = OUT / "_tts_tmp.wav"
    texts = {
        "voice_speech_en": "The quick brown fox jumps over the lazy dog. Peter Piper picked a peck of pickled peppers, but time and tide wait for no man.",
        "voice_speech_ja": "音声のピッチシフトとタイムストレッチの品質を評価します。子音と母音のフォルマントが自然に保たれるかを確認してください。",
    }
    for name, text in texts.items():
        script = ps.replace("{out}", str(tmp)).replace("{text}", text)
        r = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                           capture_output=True, text=True)
        if r.returncode != 0 or not tmp.exists():
            print(f"  !! TTS failed for {name}: {r.stderr[:200]}", file=sys.stderr)
            continue
        x, sr = sf.read(tmp, dtype="float32")
        save(name, x, sr)
        tmp.unlink(missing_ok=True)


# ------------------------------------------------------------- sung vowel
def glottal_pulse_train(f0_curve, sr=SR):
    """Rosenberg-ish glottal source following an f0 curve (Hz per sample)."""
    phase = np.cumsum(f0_curve / sr) % 1.0
    # Rosenberg pulse: rising 0..0.4, falling 0.4..0.6, closed after
    g = np.zeros_like(phase)
    up = phase < 0.4
    g[up] = 0.5 * (1 - np.cos(np.pi * phase[up] / 0.4))
    dn = (phase >= 0.4) & (phase < 0.6)
    g[dn] = np.cos(np.pi * (phase[dn] - 0.4) / 0.4)
    return np.diff(g, prepend=0.0)  # derivative -> richer spectrum


def formant_filter(x, formants, bws, gains, sr=SR):
    from scipy.signal import lfilter
    y = np.zeros_like(x)
    for f, bw, g in zip(formants, bws, gains):
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * f / sr
        a = [1, -2 * r * np.cos(theta), r * r]
        y += g * lfilter([1 - r], a, x)
    return y


def gen_sung_vowel():
    dur = 5.0
    n = int(dur * SR)
    t = np.arange(n) / SR
    # A3 with vibrato + slight jitter, gliding to E4 mid-way
    f0 = np.where(t < 2.5, 220.0, 220.0 * 2 ** (np.clip((t - 2.5) / 0.3, 0, 1) * 7 / 12))
    f0 = f0 * (1 + 0.018 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(t / 1.0, 1))
    f0 = f0 * (1 + 0.002 * rng.standard_normal(n).cumsum() / np.sqrt(np.arange(1, n + 1)))
    src = glottal_pulse_train(f0)
    # vowel /a/ -> /i/ morph in second half
    morph = np.clip((t - 2.5) / 1.5, 0, 1)
    y = np.zeros(n)
    seg = 4800
    for i in range(0, n, seg):
        m = morph[min(i + seg // 2, n - 1)]
        F = [(1 - m) * 800 + m * 300, (1 - m) * 1200 + m * 2300, 2500, 3500]
        y[i:i + seg] = formant_filter(src[i:i + seg], F, [80, 100, 120, 150], [1, 0.6, 0.35, 0.2])
    breath = formant_filter(rng.standard_normal(n) * 0.02, [2500, 3500], [800, 900], [1, 0.7])
    y = y + breath * adsr(n, 0.3, 0.5, 0.8, 0.5)
    save("voice_sung_vowel", y * adsr(n, 0.05, 0.1, 0.95, 0.2))


# ---------------------------------------------------------------- drums
def kick(n):
    t = np.arange(n) / SR
    f = 150 * np.exp(-t * 18) + 45
    ph = 2 * np.pi * np.cumsum(f) / SR
    click = np.exp(-t * 900)
    return (np.sin(ph) * np.exp(-t * 7) + 0.4 * click * rng.standard_normal(n) * np.exp(-t * 400))


def snare(n):
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 25)
    noise = rng.standard_normal(n) * np.exp(-t * 18)
    from scipy.signal import butter, lfilter
    b, a = butter(2, 1800 / (SR / 2), "high")
    return 0.5 * tone + 0.8 * lfilter(b, a, noise)


def hat(n, open_=False):
    t = np.arange(n) / SR
    from scipy.signal import butter, lfilter
    b, a = butter(4, 7000 / (SR / 2), "high")
    return lfilter(b, a, rng.standard_normal(n)) * np.exp(-t * (8 if open_ else 60))


def gen_drums():
    bpm = 118
    step = int(SR * 60 / bpm / 4)  # 16th
    pat_k = [0, 7, 8, 10]
    pat_s = [4, 12]
    pat_h = range(0, 16, 2)
    bars, y = 2, np.zeros(step * 16 * 2 + SR)
    events = set()
    for bar in range(bars):
        off = bar * step * 16
        for p in pat_k:
            i = off + p * step
            y[i:i + step * 3] += kick(step * 3)
            events.add(i)
        for p in pat_s:
            i = off + p * step
            y[i:i + step * 2] += snare(step * 2)
            events.add(i)
        for j, p in enumerate(pat_h):
            i = off + p * step
            y[i:i + step] += 0.35 * hat(step, open_=(j % 4 == 3))
            events.add(i)
    save("drums_loop", y[:step * 16 * bars + int(0.4 * SR)])
    TRUTH["drums_loop"] = {"onsets": sorted(i / SR for i in events)}
    # castanet-style sparse clicks (classic transient torture test)
    n = int(3.0 * SR)
    c = np.zeros(n)
    clicks = [0.3, 0.9, 1.5, 1.65, 2.1, 2.7]
    for pos in clicks:
        i = int(pos * SR)
        ln = int(0.03 * SR)
        t = np.arange(ln) / SR
        c[i:i + ln] += np.sin(2 * np.pi * 2200 * t) * np.exp(-t * 300) + \
                       0.5 * rng.standard_normal(ln) * np.exp(-t * 500)
    save("clicks_castanet", c)
    TRUTH["clicks_castanet"] = {"onsets": clicks}


# ---------------------------------------------------------------- pluck / piano
def karplus(f0, dur, bright=0.5):
    n = int(dur * SR)
    p = int(SR / f0)
    buf = rng.standard_normal(p) * (1 - bright) + rng.uniform(-1, 1, p) * bright
    out = np.zeros(n)
    for i in range(n):
        out[i] = buf[i % p]
        buf[i % p] = 0.997 * 0.5 * (buf[i % p] + buf[(i + 1) % p])
    return out


def gen_pluck():
    chord = [82.41, 123.47, 164.81, 207.65, 246.94, 329.63]  # E major-ish
    y = np.zeros(int(5.5 * SR))
    onsets = []
    for j, f in enumerate(chord):
        i = int(j * 0.18 * SR)
        note = karplus(f, 3.5)
        y[i:i + len(note)] += note * 0.5
        onsets.append(i / SR)
    i = int(2.2 * SR)
    for j, f in enumerate(chord):  # strum
        k = i + int(j * 0.012 * SR)
        note = karplus(f, 3.0)
        y[k:k + len(note)] += note * 0.5
    # NOTE: the strum (6 staggered notes over ~60 ms) is excluded from truth:
    # detectors legitimately report 1-3 peaks for it, so it would only inject
    # matching noise into precision/recall.
    save("pluck_guitar", y)
    TRUTH["pluck_guitar"] = {"onsets": onsets}


def piano_note(f0, dur, vel=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    B = 0.0004  # inharmonicity
    for k in range(1, 18):
        fk = f0 * k * np.sqrt(1 + B * k * k)
        if fk > SR / 2 - 2000:
            break
        amp = vel * np.exp(-k * 0.55) * np.exp(-t * (2.5 + 0.4 * k))
        y += amp * np.sin(2 * np.pi * fk * t + rng.uniform(0, 2 * np.pi))
    ham = np.exp(-t * 250) * rng.standard_normal(n) * 0.08 * vel
    return (y + ham) * adsr(n, 0.002, 0.1, 0.8, 0.3)


def gen_piano():
    events = [(0.0, [261.63, 329.63, 392.0]), (1.2, [220.0, 277.18, 329.63]),
              (2.4, [174.61, 220.0, 261.63]), (3.6, [196.0, 246.94, 293.66, 392.0])]
    y = np.zeros(int(6.5 * SR))
    for t0, notes in events:
        for f in notes:
            i = int(t0 * SR)
            nt = piano_note(f, 2.5, 0.9)
            end = min(i + len(nt), len(y))
            y[i:end] += nt[:end - i]
    save("piano_chords", y)
    TRUTH["piano_chords"] = {"onsets": [t for t, _ in events]}


# ---------------------------------------------------------------- pad / mix
def gen_pad():
    dur, n = 6.0, int(6.0 * SR)
    t = np.arange(n) / SR
    L = np.zeros(n)
    R = np.zeros(n)
    chord = [110.0, 164.81, 220.0, 277.18, 329.63]
    for f in chord:
        for det, pan in [(-0.15, 0.2), (0.0, 0.5), (0.12, 0.8)]:
            ff = f * 2 ** (det / 12 / 4)
            saw = np.zeros(n)
            for k in range(1, int(9000 / ff)):
                saw += np.sin(2 * np.pi * ff * k * t + rng.uniform(0, 6.28)) / k
            saw *= 0.08
            L += saw * (1 - pan)
            R += saw * pan
    from scipy.signal import butter, lfilter
    b, a = butter(2, (1200 + 800 * np.sin(1)) / (SR / 2))
    env = adsr(n, 1.2, 0.5, 0.9, 1.5)
    save("strings_pad", np.vstack([lfilter(b, a, L) * env, lfilter(b, a, R) * env]))


def gen_mix():
    # bass + pad + drums + arp: "full mix" stereo
    n = int(6.0 * SR)
    t = np.arange(n) / SR
    bass_f = np.where((t % 4) < 2, 55.0, 41.2)
    bass = np.sin(2 * np.pi * np.cumsum(bass_f) / SR)
    bass += 0.3 * np.sin(4 * np.pi * np.cumsum(bass_f) / SR)
    gate = ((t * 2) % 1 < 0.8).astype(float)
    bass *= gate * 0.5
    bpm, step = 118, int(SR * 60 / 118 / 4)
    dr = np.zeros(n)
    mix_events = set()
    for bar in range(4):
        off = bar * step * 16
        for p in [0, 8, 10]:
            i = off + p * step
            if i + step * 3 < n:
                dr[i:i + step * 3] += kick(step * 3)
                mix_events.add(i)
        for p in [4, 12]:
            i = off + p * step
            if i + step * 2 < n:
                dr[i:i + step * 2] += snare(step * 2)
                mix_events.add(i)
        for p in range(0, 16, 2):
            i = off + p * step
            if i + step < n:
                # hats buried under bass/pad/arp are NOT reliably detectable
                # perceptual onsets -> excluded from mix ground truth
                dr[i:i + step] += 0.3 * hat(step)
    arpL, arpR = np.zeros(n), np.zeros(n)
    notes = [440.0, 554.37, 659.26, 880.0]
    for j in range(int(6.0 * 4)):
        i = int(j * 0.25 * SR)
        f = notes[j % 4]
        ln = int(0.22 * SR)
        if i + ln > n:
            break
        tt = np.arange(ln) / SR
        nt = np.sin(2 * np.pi * f * tt) * np.exp(-tt * 12) * 0.25
        (arpL if j % 2 else arpR)[i:i + ln] += nt
    L = bass + dr + arpL + 0.7 * arpR
    R = bass + dr + arpR + 0.7 * arpL
    save("full_mix", np.vstack([L, R]))
    TRUTH["full_mix"] = {"onsets": sorted(i / SR for i in mix_events)}


# ---------------------------------------------------------------- analytic
def gen_analytic():
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    save("tone_sine_440", np.sin(2 * np.pi * 440 * t))
    saw = np.zeros(n)
    for k in range(1, 200):
        f = 110 * k
        if f > 20000:
            break
        saw += np.sin(2 * np.pi * f * t) / k
    save("tone_harmonic_A2", saw)
    save("sweep_log", np.sin(2 * np.pi * 20 * (np.exp(t / 3.0 * np.log(20000 / 20)) - 1)
                             * 3.0 / np.log(20000 / 20)))
    am = np.sin(2 * np.pi * 1000 * t) * (1 + 0.5 * np.sin(2 * np.pi * 4 * t))
    save("tone_am_4hz", am)  # modulation-spectrum reference


if __name__ == "__main__":
    print(f"Generating corpus -> {OUT}")
    gen_tts_speech()
    gen_sung_vowel()
    gen_drums()
    gen_pluck()
    gen_piano()
    gen_pad()
    gen_mix()
    gen_analytic()
    (OUT / "ground_truth.json").write_text(json.dumps(TRUTH, indent=1))
    print(f"ground_truth.json: {list(TRUTH)}")
    print("done")
