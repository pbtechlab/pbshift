#!/usr/bin/env python3
"""Render one signal across several time-stretch ratios, level-matched for A/B.

alpha = output length / input length, so alpha < 1 shortens and alpha > 1
lengthens. Both directions are swept deliberately: they fail in different ways
(lengthening duplicates material, shortening collides transients), and an engine
that holds together at 2x can still come apart at 0.5x.

Engine-agnostic, like the rest of `benchmarks/`. pbshift's own modes are always
rendered; any other engine following the shared CLI convention

    engine in.wav out.wav --stretch <ratio>

can be added through PBSHIFT_REFERENCES as a comma-separated list of
`name=path` pairs. The repository ships only pbshift.

Two fairness rules are applied, because without them the sweep measures the
wrong thing:

  headroom  -- a full-scale master is attenuated to -6 dBFS before it reaches
               ANY engine. Stretching a 0 dBFS mix can overshoot, and an engine
               that clips internally would be scored on the source level rather
               than on its algorithm.
  loudness  -- every render is normalised to one integrated loudness, because a
               listening comparison is decided by level long before it is
               decided by quality.

usage:
  python benchmarks/ratio_sweep.py in.wav [alpha ...]
  PBSHIFT_REFERENCES="other=/path/other.exe" python benchmarks/ratio_sweep.py in.wav
"""
import os
import subprocess
import sys

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Sample")
PBSHIFT = os.environ.get("PBSHIFT_CLI",
                         os.path.join(ROOT, "tools", "bin", "pbshift.exe"))
TARGET_LUFS = -21.0
HEADROOM_PEAK = 0.5                      # -6 dBFS
RATIOS = [0.5, 0.8, 1.25, 2.0, 4.0]

PB_MODES = [("pbshift", []), ("pbshift_multi", ["--multi"]),
            ("pbshift_rhythm", ["--rhythm"]), ("pbshift_voice", ["--voice"])]


def references():
    """name -> executable, parsed from PBSHIFT_REFERENCES ("a=path,b=path")."""
    out = {}
    for item in filter(None, (s.strip() for s in
                              os.environ.get("PBSHIFT_REFERENCES", "").split(","))):
        if "=" in item:
            name, path = item.split("=", 1)
            out[name.strip()] = path.strip()
    return out


def tag_of(alpha):
    return ("%g" % alpha) + "x"


def with_headroom(path):
    """Give every engine the same headroom (see module docstring)."""
    x, sr = sf.read(path)
    peak = np.abs(x).max()
    if peak <= HEADROOM_PEAK + 0.01:
        return path
    src = os.path.join(OUT, "_src_" + os.path.basename(path))
    sf.write(src, x * (HEADROOM_PEAK / peak), sr, subtype="PCM_16")
    return src


def normalize(path):
    x, sr = sf.read(path)
    mono = x.mean(axis=1) if x.ndim > 1 else x
    meter = pyln.Meter(sr)
    y = pyln.normalize.loudness(x, meter.integrated_loudness(mono), TARGET_LUFS)
    peak = np.abs(y).max()
    if peak > 0.99:                      # keep the level match, avoid clipping
        y = y * (0.99 / peak)
    sf.write(path, y, sr, subtype="PCM_16")


def render(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def main():
    args = sys.argv[1:]
    if not args:
        print("usage: python benchmarks/ratio_sweep.py in.wav [alpha ...]")
        return
    inp = with_headroom(os.path.abspath(args[0]))
    name = os.path.splitext(os.path.basename(args[0]))[0]
    ratios = [float(a) for a in args[1:]] or RATIOS

    engines = [(n, [PBSHIFT] + flags) for n, flags in PB_MODES]
    engines += [(n, [p]) for n, p in references().items()]

    os.makedirs(OUT, exist_ok=True)
    for alpha in ratios:
        for eng, head in engines:
            outp = os.path.join(OUT, f"{name}_{tag_of(alpha)}_{eng}.wav")
            cmd = [head[0], inp, outp, "--stretch", str(alpha)] + head[1:]
            try:
                render(cmd)
                normalize(outp)
                print(f"ok   {os.path.basename(outp):44} "
                      f"{sf.info(outp).duration:7.3f}s")
            except Exception as exc:                       # noqa: BLE001
                print(f"FAIL {os.path.basename(outp):44} {exc}")


if __name__ == "__main__":
    main()
