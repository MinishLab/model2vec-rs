#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""Draw the throughput of Rust, the Python bindings and Python model2vec.

Takes the raw JSON results of a `python_compare.py` run with `--bindings` and
`--threads 1,all`, and writes `assets/images/bindings_throughput.png`.

Usage (from the repository root):

    uv run --with ./python scripts/python_compare.py --bindings --rust-api flat --no-python-multiprocessing --out bindings.json
    uv run scripts/bindings_chart.py bindings.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b5b4ae"
GROUPS = [("1", "Single thread"), ("all", "All cores")]


def fmt_rate(x: float) -> str:
    return f"{x / 1e6:.2f}M/s" if x >= 1e6 else f"{x / 1e3:.0f}K/s"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path, help="python_compare.py results with --bindings")
    p.add_argument("--hardware", default="Apple M3 Pro")
    p.add_argument("--out", type=Path, default=REPO / "assets" / "images" / "bindings_throughput.png")
    a = p.parse_args()

    data = json.loads(a.results.read_text())
    meta, results = data["meta"], data["results"]
    rust_method = "encode_flat" if meta.get("rust_api") == "flat" else "encode"
    series = [
        ("rust", f"Rust ({rust_method})", "#2a78d6"),
        ("bindings", "Python bindings", "#eb6834"),
        ("python", "Python model2vec", "#1baf7a"),
    ]

    plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 12})
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    height, gap = 0.24, 0.02
    top = max(results[t]["median_texts_per_second"][side] for t, _ in GROUPS for side, _, _ in series)
    for i, (side, name, color) in enumerate(series):
        for g, (threads, _) in enumerate(GROUPS):
            rate = results[threads]["median_texts_per_second"][side]
            y = g + (i - 1) * (height + gap)
            ax.barh(y, rate, height=height, color=color, label=name if g == 0 else None)
            ax.text(rate + top * 0.012, y, fmt_rate(rate), va="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_yticks(range(len(GROUPS)), [label for _, label in GROUPS], fontsize=13, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, top * 1.15)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:g}M" if x else "0")
    ax.tick_params(axis="x", colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.legend(loc="upper right", frameon=False, fontsize=11, labelcolor=INK)

    model = meta["model"].split("/")[-1]
    corpus = Path(meta["corpus"]).stem
    words = corpus.removeprefix("wiki-en-")
    size = f" ({int(words) / 1e6:g}M words)" if words.isdigit() else ""
    fig.text(0.015, 0.94, "Encoding throughput: Rust, Python bindings and model2vec", fontsize=17, fontweight="bold", color=INK)
    fig.text(
        0.015,
        0.88,
        f"{model}, {meta['texts']:,} Wikipedia paragraphs{size}, {a.hardware}. Texts per second, higher is better.",
        fontsize=10,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.15, right=0.98, top=0.82, bottom=0.08)
    fig.savefig(a.out, facecolor=SURFACE)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
