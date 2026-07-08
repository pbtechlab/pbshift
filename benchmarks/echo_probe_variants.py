#!/usr/bin/env python3
"""Run echo_probe metrics over the pbshift fix candidates too.

Reuses echo_probe's analysis functions; just extends the FILES list with the
new candidate renders so pbshift(default) / voice / coherent are compared
head-to-head against Bungee and the input.
"""
import echo_probe as ep

ep.FILES = [
    ("input",       "Sample/tda_0121_clm_n_f.wav"),
    ("pb_default",  "Sample/tda_0121_2x_pbshift.wav"),
    ("pb_voice",    "Sample/tda_0121_2x_pbshift_voice.wav"),
    ("pb_coherent", "Sample/tda_0121_2x_pbshift_coherent.wav"),
    ("Bungee",      "Sample/tda_0121_2x_Bungee.wav"),
    ("Signalsmith", "Sample/tda_0121_2x_Signalsmith.wav"),
    ("RubberBand",  "Sample/tda_0121_2x_RubberBand.wav"),
]

# the differential block hard-codes rows["pbshift"]; alias it so it still runs
_orig_analyze = ep.analyze


def analyze():
    sr, pitch_ms, pitch_val, rows = _orig_analyze()
    rows["pbshift"] = rows["pb_default"]
    return sr, pitch_ms, pitch_val, rows


ep.analyze = analyze

if __name__ == "__main__":
    ep.main()
