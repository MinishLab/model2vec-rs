#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""Draw the README speedup chart from `python_compare.py` results.

Takes the raw JSON results of a float32 and an int8 run, both with
`--threads 1,all`, and writes `assets/images/speedup.png`. Model load is the
median over every process launch of a run, across both thread settings.

Usage (from the repository root):

    uv run scripts/python_compare.py --words 11000000 --no-python-multiprocessing --out f32.json
    uv run scripts/python_compare.py --words 11000000 --no-python-multiprocessing --quantize int8 --out int8.json
    uv run scripts/speedup_chart.py f32.json int8.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b5b4ae"
GROUPS = ["Single thread", "All cores", "Model load"]


def fmt_rate(x: float) -> str:
    return f"{x / 1e6:.2f}M/s" if x >= 1e6 else f"{x / 1e3:.0f}K/s"


def speedups(path: Path) -> tuple[dict, list[float], list[str]]:
    data = json.loads(path.read_text())
    results = data["results"]
    values, notes = [], []
    for threads in ("1", "all"):
        rate = results[threads]["median_texts_per_second"]
        values.append(rate["rust"] / rate["python"])
        notes.append(f"{fmt_rate(rate['rust'])} vs {fmt_rate(rate['python'])}")
    load = {side: statistics.median(results["1"][side]["load"] + results["all"][side]["load"]) for side in ("rust", "python")}
    values.append(load["python"] / load["rust"])
    notes.append(f"{load['rust'] * 1e3:.0f} ms vs {load['python'] * 1e3:.0f} ms")
    return data["meta"], values, notes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("float32", type=Path, help="python_compare.py results without --quantize")
    p.add_argument("int8", type=Path, help="python_compare.py results with --quantize int8")
    p.add_argument("--hardware", default="Apple M3 Pro")
    p.add_argument("--out", type=Path, default=REPO / "assets" / "images" / "speedup.png")
    a = p.parse_args()

    series = [("float32", "#2a78d6", *speedups(a.float32)), ("int8", "#eb6834", *speedups(a.int8))]
    meta = series[0][2]

    plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 12})
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    height, gap = 0.34, 0.02
    for i, (name, color, _, values, notes) in enumerate(series):
        for g, (value, note) in enumerate(zip(values, notes)):
            y = g + (i - 0.5) * (height + gap)
            ax.barh(y, value, height=height, color=color, label=name if g == 0 else None)
            ax.text(value + 0.4, y, f"{value:.1f}×", va="center", fontsize=12, fontweight="bold", color=INK)
            ax.text(value + 4.0, y, note, va="center", fontsize=10, color=MUTED)
    ax.set_yticks(range(len(GROUPS)), GROUPS, fontsize=13, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 40)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}×")
    ax.tick_params(axis="x", colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.legend(loc="lower right", frameon=False, fontsize=12, labelcolor=INK)

    model = meta["model"].split("/")[-1]
    corpus = Path(meta["corpus"]).stem
    words = corpus.removeprefix("wiki-en-")
    size = f" ({int(words) / 1e6:g}M words)" if words.isdigit() else ""
    fig.text(0.015, 0.95, "model2vec-rs speedup over Python model2vec", fontsize=17, fontweight="bold", color=INK)
    fig.text(
        0.015,
        0.895,
        f"{model}, {meta['texts']:,} Wikipedia paragraphs{size}, {a.hardware}. Higher is better; gray labels show Rust vs Python.",
        fontsize=10,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.15, right=0.98, top=0.84, bottom=0.07)
    fig.savefig(a.out, facecolor=SURFACE)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
