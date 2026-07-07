# -*- coding: utf-8 -*-
"""
Win/loss dashboard: aggregates out/results.csv into per-class, per-metric
engine comparisons and an overall scorecard vs each baseline.

Usage: python report.py [results.csv] > out/report.md
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "out" / "results.csv"

# metric -> higher_is_better
METRICS = {
    "hnr_db": True, "onset_recall": True, "onset_precision": True,
    "hnr_drop_db": False, "attack_ratio_dev": False, "duration_err": False,
    "f0_err_cents_abs": False, "envelope_lsd_db": False,
    "ltas_dist_db": False, "warble_db": False, "stereo_coh_drift_abs": False,
}

# audibility-based tie rules: (absolute difference floor, "both beyond
# audibility" saturation point or None). Differences smaller than the floor,
# or comparisons where both engines are already past the saturation point,
# are inaudible and count as ties.
TIE = {
    "hnr_db": (3.0, 55.0),            # both >= 55 dB -> inaudible either way
    "hnr_drop_db": (3.0, None),
    "onset_recall": (0.03, None),
    "onset_precision": (0.03, None),
    "attack_ratio_dev": (0.10, None),
    "duration_err": (0.005, None),
    "f0_err_cents_abs": (1.0, None),  # sub-cent differences are inaudible
    "envelope_lsd_db": (0.30, None),
    "ltas_dist_db": (0.15, None),
    "warble_db": (2.0, -40.0),        # both <= -40 dB -> inaudible
    "stereo_coh_drift_abs": (0.02, None),
}

CLASS_OF = {}  # filled from rows


def norm_row(r):
    out = {}
    for k, v in r.items():
        if v in (None, ""):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if k == "attack_ratio":
            out["attack_ratio_dev"] = abs(f - 1.0)  # 1.0 = perfect
        elif k == "f0_err_cents":
            out["f0_err_cents_abs"] = abs(f)
        elif k == "stereo_coh_drift":
            out["stereo_coh_drift_abs"] = abs(f)
        elif k in METRICS:
            out[k] = f
    return out


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    # values[(cls, metric)][engine] = list
    values = defaultdict(lambda: defaultdict(list))
    cond_vals = defaultdict(dict)  # (signal,pitch,stretch,metric) -> {engine: val}
    engines = []
    errors = defaultdict(int)
    for r in rows:
        e = r["engine"]
        if e not in engines:
            engines.append(e)
        if r.get("error"):
            errors[e] += 1
            continue
        cls = r["class"]
        m = norm_row(r)
        for k, v in m.items():
            values[(cls, k)][e].append(v)
            cond = (r["signal"], r["pitch"], r["stretch"], k)
            cond_vals[cond][e] = v

    print("# Benchmark dashboard\n")
    print(f"source: {CSV.name}, rows: {len(rows)}, engines: {', '.join(engines)}")
    if errors:
        print(f"render errors: {dict(errors)}")

    # ---- per-class medians table -----------------------------------
    classes = sorted({c for (c, _) in values})
    for cls in classes:
        print(f"\n## class: {cls}\n")
        mets = sorted({k for (c, k) in values if c == cls})
        hdr = "| metric | " + " | ".join(engines) + " | winner |"
        print(hdr)
        print("|" + "---|" * (len(engines) + 2))
        for k in mets:
            row = values[(cls, k)]
            meds = {e: float(np.median(row[e])) for e in engines if row.get(e)}
            if not meds:
                continue
            hib = METRICS[k]
            best = max(meds, key=lambda e: meds[e] if hib else -meds[e])
            cells = " | ".join(
                f"**{meds[e]:.3f}**" if e == best else f"{meds[e]:.3f}"
                if e in meds else "-" for e in engines)
            print(f"| {k} | {cells} | {best} |")

    # ---- head-to-head: pb vs each baseline, per condition -----------
    if "pb" in engines:
        print("\n## head-to-head (pb vs baseline, per signal x condition x metric)\n")
        print("| baseline | pb wins | ties(<2%) | losses | win rate |")
        print("|---|---|---|---|---|")
        for base in engines:
            if base == "pb":
                continue
            w = t = l = 0
            for cond, ev in cond_vals.items():
                if "pb" not in ev or base not in ev:
                    continue
                k = cond[3]
                a, b = ev["pb"], ev[base]
                hib = METRICS[k]
                floor_, sat = TIE.get(k, (0.0, None))
                saturated = False
                if sat is not None:
                    if hib:
                        saturated = a >= sat and b >= sat
                    else:
                        saturated = a <= sat and b <= sat
                if saturated or abs(a - b) < floor_:
                    t += 1
                elif (a > b) == hib:
                    w += 1
                else:
                    l += 1
            tot = w + t + l
            print(f"| {base} | {w} | {t} | {l} | "
                  f"{100.0 * (w + 0.5 * t) / max(tot, 1):.1f}% |")


if __name__ == "__main__":
    main()
