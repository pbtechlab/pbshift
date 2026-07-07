# -*- coding: utf-8 -*-
"""
Benchmark runner: renders corpus x conditions through every available engine
CLI and aggregates objective metrics.

Engine CLI convention:
    engine.exe in.wav out.wav --pitch <semitones> --stretch <ratio> [--formant]
(stretch = output_duration / input_duration)

Usage:
    python run_bench.py                # all engines found in tools/bin
    python run_bench.py --engines pb --quick
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from metrics import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path(__file__).parent / "corpus"
OUTDIR = Path(__file__).parent / "out"
BIN = ROOT / "tools" / "bin"

# name -> (exe, supports_formant_flag). Engine-agnostic: our engine plus any
# comparison adapter you drop into tools/bin/ as `baseline_<name>.exe` (each must
# accept the shared CLI convention). The adapters themselves are local-only and
# not part of this repository; only pbshift ships here.
ENGINE_CANDIDATES = {
    "pb": (BIN / "pbshift.exe", True),  # ours
}
for _exe in sorted(BIN.glob("baseline_*.exe")):
    ENGINE_CANDIDATES.setdefault(_exe.stem[len("baseline_"):], (_exe, True))

# signal -> (class, f0_or_None)
SIGNALS = {
    "voice_speech_en": ("voice", None),
    "voice_speech_ja": ("voice", None),
    "voice_sung_vowel": ("voice", None),
    "drums_loop": ("percussive", None),
    "clicks_castanet": ("percussive", None),
    "pluck_guitar": ("mix", None),
    "piano_chords": ("mix", None),
    "strings_pad": ("pad", None),
    "full_mix": ("mix", None),
    "tone_sine_440": ("tonal", 440.0),
    "tone_harmonic_A2": ("tonal", 110.0),
    "tone_am_4hz": ("tonal", 1000.0),
}

STRETCH_ONLY = [0.25, 0.5, 0.8, 1.25, 2.0, 4.0]
PITCH_ONLY = [-24, -12, -5, 5, 12, 24]
COMBOS = [(7, 1.5), (-7, 0.75)]

QUICK_STRETCH = [0.5, 2.0]
QUICK_PITCH = [-12, 12]


def conditions(quick=False):
    st = QUICK_STRETCH if quick else STRETCH_ONLY
    pt = QUICK_PITCH if quick else PITCH_ONLY
    for s in st:
        yield (0, s)
    for p in pt:
        yield (p, 1.0)
    if not quick:
        yield from COMBOS


def run_engine(exe, formant_ok, in_wav, out_wav, pitch, stretch, use_formant):
    cmd = [str(exe), str(in_wav), str(out_wav),
           "--pitch", str(pitch), "--stretch", str(stretch)]
    if use_formant and formant_ok:
        cmd.append("--formant")
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None, "render timeout"
    except OSError as e:
        return None, f"exec: {e}"
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        return None, r.stderr.strip()[:300]
    return dt, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default=None, help="comma list")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--signals", default=None, help="comma list")
    args = ap.parse_args()

    engines = {}
    want = args.engines.split(",") if args.engines else None
    for name, (exe, fmt) in ENGINE_CANDIDATES.items():
        if want and name not in want:
            continue
        if exe.exists():
            engines[name] = (exe, fmt)
        elif want:
            print(f"!! engine '{name}' requested but {exe} missing")
    if not engines:
        print("No engine executables found in", BIN)
        sys.exit(1)
    print("Engines:", ", ".join(engines))

    signals = {k: v for k, v in SIGNALS.items()
               if not args.signals or k in args.signals.split(",")}

    truth_path = CORPUS / "ground_truth.json"
    truth = {}
    if truth_path.exists():
        import json
        truth = json.loads(truth_path.read_text())

    rows = []
    conds = list(conditions(args.quick))
    total = len(engines) * len(signals) * len(conds)
    done = 0
    for sig, (cls, f0) in signals.items():
        in_wav = CORPUS / f"{sig}.wav"
        if not in_wav.exists():
            print(f"skip missing {in_wav}")
            continue
        for eng, (exe, fmt_ok) in engines.items():
            odir = OUTDIR / eng
            odir.mkdir(parents=True, exist_ok=True)
            for pitch, stretch in conds:
                done += 1
                tag = f"{sig}__p{pitch:+d}_s{stretch}".replace(".", "_")
                out_wav = odir / f"{tag}.wav"
                use_fmt = cls == "voice" and pitch != 0
                dt, err = run_engine(exe, fmt_ok, in_wav, out_wav,
                                     pitch, stretch, use_fmt)
                row = {"engine": eng, "signal": sig, "class": cls,
                       "pitch": pitch, "stretch": stretch,
                       "render_s": round(dt, 3) if dt else None, "error": err}
                if err is None and out_wav.exists():
                    try:
                        m = evaluate(in_wav, out_wav, stretch, pitch, cls,
                                     f0_in=f0, formant_mode=use_fmt and fmt_ok,
                                     truth_onsets=truth.get(sig, {}).get("onsets"))
                        row.update({k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in m.items()})
                    except Exception as e:  # metric failure shouldn't kill run
                        row["error"] = f"metric: {e}"
                print(f"[{done}/{total}] {eng:12s} {tag:44s} "
                      f"{'ERR ' + str(row['error'])[:60] if row['error'] else 'ok'}")
                rows.append(row)

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    OUTDIR.mkdir(exist_ok=True)
    with open(OUTDIR / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUTDIR / 'results.csv'} ({len(rows)} rows)")
    summarize(rows)


def summarize(rows):
    """Per-engine medians of each metric, worse-is-higher normalized."""
    import collections
    metric_dirs = {  # +1 higher=better, -1 lower=better
        "onset_recall": 1, "onset_precision": 1, "hnr_db": 1,
        "attack_ratio": -1, "duration_err": -1, "hnr_drop_db": -1,
        "f0_err_cents_abs": -1, "envelope_lsd_db": -1, "ltas_dist_db": -1,
        "warble_db": -1, "stereo_coh_drift": -1,
    }
    by_engine = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("error"):
            continue
        for k, v in r.items():
            if v is None:
                continue
            if k == "f0_err_cents":
                by_engine[r["engine"]]["f0_err_cents_abs"].append(abs(v))
            elif k in metric_dirs:
                by_engine[r["engine"]][k].append(v)
    print("\n=== per-engine medians ===")
    metrics_present = sorted({m for e in by_engine.values() for m in e})
    hdr = "metric".ljust(20) + "".join(e.rjust(14) for e in by_engine)
    print(hdr)
    for m in metrics_present:
        line = m.ljust(20)
        for e in by_engine:
            vals = by_engine[e].get(m, [])
            line += (f"{np.median(vals):.3f}".rjust(14) if vals else "-".rjust(14))
        print(line)
    fails = collections.Counter(r["engine"] for r in rows if r.get("error"))
    if fails:
        print("\nrender/metric errors:", dict(fails))


if __name__ == "__main__":
    main()
