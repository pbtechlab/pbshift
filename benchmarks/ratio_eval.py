#!/usr/bin/env python3
"""Render pb-multires and a reference engine for one corpus signal across
several time-stretch ratios and print flutter / transient / integrity for each,
so a reviewer can judge win/loss per ratio.

usage: ratio_eval.py <signal-basename> [--voice]
  --voice : route pb through the single-long-window voiced layout (for held
            sung vowels / sustained tones).

The reference engine is whatever executable is at $PBSHIFT_BASELINE (default
tools/bin/baseline.exe); it just needs the shared CLI convention
`engine in.wav out.wav --stretch <ratio>`. Engine-agnostic: point it at any
build you want to compare against.

Deterministic (no randomness). Renders to benchmarks/chorus/re_*.wav.
"""
import subprocess
import sys
import os

sys.path.insert(0, "benchmarks")
from eval_one import measure, transient_sharp, band_energies, load  # noqa: E402

RATIOS = [0.5, 0.8, 1.25, 2.0, 4.0]
MR = os.path.abspath("tools/bin/multires.exe")
REF = os.path.abspath(os.environ.get("PBSHIFT_BASELINE", "tools/bin/baseline.exe"))
CORPUS = "benchmarks/corpus"
OUT = "benchmarks/chorus"


def render(engine, inp, outp, ratio, voice):
    cmd = [engine, inp, outp, "--stretch", str(ratio)]
    if engine == MR and voice:
        cmd.append("--voice")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def metrics(path, src, ssr):
    x, sr = load(path)
    m = measure(path) or {"flutter_db": float("nan")}
    ob = band_energies(x, sr)
    sb = band_energies(src, ssr)
    dmid = (ob[1] - ob[0]) - (sb[1] - sb[0])
    dhi = (ob[2] - ob[0]) - (sb[2] - sb[0])
    degen = dmid < -20 or dhi < -20
    return m["flutter_db"], transient_sharp(x), ("DEGEN" if degen else "OK")


def main():
    sig = sys.argv[1]
    voice = "--voice" in sys.argv[2:]
    src, ssr = load(f"{CORPUS}/{sig}.wav")
    print(f"# {sig}  (pb {'VOICED' if voice else 'default'} vs the reference engine)")
    print(f"# {'ratio':>6} | {'pb_flutter':>10} {'pb_trans':>9} {'pb_int':>6} "
          f"| {'ref_flutter':>10} {'ref_trans':>9} {'ref_int':>6}")
    for r in RATIOS:
        pp = f"{OUT}/re_{sig}_pb_{r}.wav"
        rp = f"{OUT}/re_{sig}_rb_{r}.wav"
        render(MR, f"{CORPUS}/{sig}.wav", pp, r, voice)
        render(REF, f"{CORPUS}/{sig}.wav", rp, r, voice)
        pf, pt, pi = metrics(pp, src, ssr)
        rf, rt, ri = metrics(rp, src, ssr)
        print(f"  {r:>6} | {pf:>+10.2f} {pt:>9.1f} {pi:>6} "
              f"| {rf:>+10.2f} {rt:>9.1f} {ri:>6}")


if __name__ == "__main__":
    main()
