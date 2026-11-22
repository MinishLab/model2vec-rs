#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""Draw the README ASCII vs non-ASCII chart from `bench_suite.py` results.

Takes the `results.json` of a `libs` run that includes model2vec-rs,
go-potion and encoders, and writes `assets/images/ascii_vs_non_ascii.png`:
throughput of each library per corpus, starting with the NanoMIRACL English
passages with every non-ASCII char removed next to the same passages as
published. It also prints the numbers as a Markdown table.

Usage (from the repository root):

    uv run scripts/bench_suite.py --sections libs --models potion-base-8M --modes single \\
        --corpora en_ascii,en,de,ru,ar,hi,zh,ja,ko --rounds 3
    uv run scripts/ascii_chart.py target/bench-suite/results/<timestamp>/results.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b5b4ae"
LIBRARIES = [("model2vec-rs", "rs:", "#2a78d6"), ("go-potion", "go-potion", "#eb6834"), ("encoders", "encoders", "#1baf7a")]
NAMES = {
    "big": "big.txt",
    "en_ascii": "English,\nASCII only",
    "en": "English,\nas published",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "fi": "Finnish",
    "id": "Indonesian",
    "sw": "Swahili",
    "yo": "Yoruba",
    "ru": "Russian",
    "ar": "Arabic",
    "fa": "Persian",
    "hi": "Hindi",
    "bn": "Bengali",
    "te": "Telugu",
    "th": "Thai",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}
MODES = {"single": "one 256-word chunk per call, one thread", "batch": "all 256-word chunks in one call, all cores"}
ENCODERS_PARALLEL_MIN, ENCODERS_MIN_LEN = 128, 32


def column(rates: dict[str, float], prefix: str) -> str | None:
    return next((k for k in rates if k.startswith(prefix) and k.endswith(" f32")), None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path, help="bench_suite.py results.json with a libs section")
    p.add_argument("--model", default="potion-base-8M")
    p.add_argument("--mode", default="single", choices=list(MODES))
    p.add_argument("--hardware", default="Apple M3 Pro")
    p.add_argument("--cores", type=int, default=os.cpu_count(), help="cores of the machine the results come from")
    p.add_argument("--out", type=Path, help="default: assets/images/ascii_vs_non_ascii[_all_cores].png")
    a = p.parse_args()
    if a.out is None:
        a.out = REPO / "assets" / "images" / ("ascii_vs_non_ascii.png" if a.mode == "single" else "ascii_vs_non_ascii_all_cores.png")

    data = json.loads(a.results.read_text())
    rows = [r for r in data["libs"] if r["model"] == a.model and r["mode"] == a.mode]
    if not rows:
        raise SystemExit(f"no libs results for {a.model} {a.mode} in {a.results}")
    min_chunks = max(ENCODERS_PARALLEL_MIN, ENCODERS_MIN_LEN * a.cores)
    dropped = [r for r in rows if a.mode == "batch" and r["chunks"] < min_chunks]
    rows = [r for r in rows if r not in dropped]
    corpora = [r["corpus"] for r in rows]
    series = []
    for name, prefix, color in LIBRARIES:
        values = []
        for r in rows:
            key = column(r["mb_per_s"], prefix)
            if key is None:
                raise SystemExit(f"{a.results} has no f32 {name} result for {r['corpus']}")
            values.append(r["mb_per_s"][key])
        series.append((name, color, values))

    print("| Corpus | " + " | ".join(name for name, _, _ in series) + " |")
    print("|---|" + "---:|" * len(series))
    for i, c in enumerate(corpora):
        print(f"| {NAMES[c].replace(chr(10), ' ')} | " + " | ".join(f"{v[i]:.0f} MB/s" for _, _, v in series) + " |")

    plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 12})
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    width, gap = 0.26, 0.02
    top = max(max(v) for _, _, v in series)
    labelled = {i for i, c in enumerate(corpora) if c in ("en_ascii", "en")}
    for s, (name, color, values) in enumerate(series):
        for g, value in enumerate(values):
            x = g + (s - 1) * (width + gap)
            ax.bar(x, value, width=width, color=color, label=name if g == 0 else None)
            if g in labelled:
                ax.text(x, value + top * 0.012, f"{value:.0f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    if "en_ascii" in corpora and "en" in corpora:
        boundary = max(corpora.index("en_ascii"), corpora.index("en")) + 0.5
        ax.axvline(boundary, color=AXIS, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_xticks(range(len(corpora)), [NAMES[c] for c in corpora], fontsize=11, color=INK)
    ax.set_ylim(0, top * 1.12)
    ax.set_ylabel("MB/s", color=MUTED, fontsize=11)
    ax.tick_params(axis="y", colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.legend(loc="upper right", frameon=False, fontsize=12, labelcolor=INK, ncols=3)

    fig.text(0.015, 0.95, "Throughput on ASCII and non-ASCII text", fontsize=17, fontweight="bold", color=INK)
    fig.text(
        0.015,
        0.895,
        f"{a.model}, float32, NanoMIRACL passages, {MODES[a.mode]}, {a.hardware}. Higher is better.",
        fontsize=10,
        color=MUTED,
    )
    if "en_ascii" in corpora and "en" in corpora:
        i, j = corpora.index("en_ascii"), corpora.index("en")
        kept = ", ".join(f"{name} {v[j] / v[i]:.0%}" for name, _, v in series)
        fig.text(
            0.015,
            0.855,
            f"English as published, against the same passages with non-ASCII chars removed, runs at: {kept} of the ASCII-only speed.",
            fontsize=10,
            color=MUTED,
        )
    bottom = 0.12
    if dropped:
        names = ", ".join(f"{NAMES[r['corpus']]} ({r['chunks']} chunks)" for r in dropped)
        fig.text(
            0.015,
            0.02,
            f"Left out: {names}, too few for encoders to use all {a.cores} cores "
            f"(it parallelizes from {ENCODERS_PARALLEL_MIN} texts, in groups of {ENCODERS_MIN_LEN} or more).",
            fontsize=9,
            color=MUTED,
        )
        bottom = 0.16
    fig.subplots_adjust(left=0.07, right=0.99, top=0.79, bottom=bottom)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=SURFACE)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
