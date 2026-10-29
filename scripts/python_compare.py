#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["model2vec==0.10.0", "numpy", "pyarrow", "huggingface_hub"]
# ///
"""Throughput comparison of model2vec-rs with Python model2vec.

Builds `benches/python_compare.rs` from this checkout, then encodes the same
texts with both implementations in separate processes, alternating Rust and
Python (R P, P R, ...) so slow drift hits both sides equally. Every non-empty
line of the corpus is one text; each timed pass encodes all of them in one
`encode` call with the default `max_length` of 512.

The default corpus is the one used for the README numbers: 2,644 random
articles from English Wikipedia (`wikimedia/wikipedia`, config `20231101.en`,
shard 17, pinned revision), shuffled with seed 0 and taken until they reach
1.1M words. It is downloaded once to `target/python-compare/wiki-en.txt` and
checked against a SHA-256 digest.

Usage (from the repository root):

    uv run scripts/python_compare.py
    uv run scripts/python_compare.py --threads 1 --rounds 3
    uv run scripts/python_compare.py --corpus my-texts.txt --model minishlab/potion-retrieval-32M

`--threads 1` sets `RAYON_NUM_THREADS=1` and `TOKENIZERS_PARALLELISM=false`
for both sides; `all` leaves both at their default parallelism. Python runs
with its default `use_multiprocessing=True` for `all`, which spreads large
inputs over a process pool; `--no-python-multiprocessing` turns that off.
Every other thread count runs Python with `use_multiprocessing=False`, since
the process pool ignores the thread limit.
The script also reports how far the Rust embeddings are from the Python ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "target" / "python-compare"
WIKI_REPO = "wikimedia/wikipedia"
WIKI_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKI_SHARD = "20231101.en/train-00017-of-00041.parquet"
WIKI_WORDS = 1_100_000
WIKI_SHA256 = "c23440073f72900784a3f56a2318742ff055506b7a136f1e473edb4ef190d9e7"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_corpus() -> Path:
    path = WORK / "wiki-en.txt"
    if path.exists() and sha256(path) == WIKI_SHA256:
        return path
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    print(f"downloading {WIKI_REPO}/{WIKI_SHARD}@{WIKI_REVISION[:12]}", file=sys.stderr)
    shard = hf_hub_download(WIKI_REPO, WIKI_SHARD, repo_type="dataset", revision=WIKI_REVISION)
    articles = pq.read_table(shard, columns=["text"]).column("text").to_pylist()
    random.Random(0).shuffle(articles)
    sample, words = [], 0
    for article in articles:
        sample.append(article)
        words += len(article.split())
        if words >= WIKI_WORDS:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("\n\n".join(sample).encode("utf-8"))
    if sha256(path) != WIKI_SHA256:
        sys.exit(f"corpus {path} does not match the expected SHA-256 {WIKI_SHA256}")
    return path


def read_texts(corpus: Path) -> list[str]:
    """Split `corpus` like `str::lines` in Rust and keep the non-empty lines."""
    lines = corpus.read_bytes().decode("utf-8").split("\n")
    return [line.removesuffix("\r") for line in lines if line.strip()]


def python_worker(corpus: Path, model_id: str, reps: int, multiprocessing: bool, out: Path | None) -> None:
    start = time.perf_counter()
    from model2vec import StaticModel

    model = StaticModel.from_pretrained(model_id)
    print(f"LOAD {time.perf_counter() - start:.6f}")
    texts = read_texts(corpus)
    print(f"TEXTS {len(texts)}")
    warm = model.encode(texts, use_multiprocessing=multiprocessing)
    if out:
        warm.astype("<f4").tofile(out)
    del warm
    for _ in range(reps):
        start = time.perf_counter()
        model.encode(texts, use_multiprocessing=multiprocessing)
        print(f"SAMPLE {time.perf_counter() - start:.9f}")


def build() -> Path:
    print("building benches/python_compare.rs", file=sys.stderr)
    out = subprocess.run(
        ["cargo", "bench", "--bench", "python_compare", "--no-run", "--message-format=json"],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    for line in out.splitlines():
        msg = json.loads(line)
        if msg.get("reason") == "compiler-artifact" and msg["target"]["name"] == "python_compare" and msg.get("executable"):
            return Path(msg["executable"])
    raise RuntimeError("could not find the python_compare executable")


def thread_env(threads: str) -> dict[str, str]:
    env = dict(os.environ)
    if threads != "all":
        env["RAYON_NUM_THREADS"] = threads
        env["RAYON_RS_NUM_THREADS"] = threads
        env["OMP_NUM_THREADS"] = threads
        if threads == "1":
            env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def run_side(cmd: list[str], threads: str) -> tuple[float, int, list[float]]:
    out = subprocess.run(cmd, env=thread_env(threads), check=True, text=True, capture_output=True).stdout
    load, texts, samples = 0.0, 0, []
    for line in out.splitlines():
        key, _, value = line.partition(" ")
        if key == "LOAD":
            load = float(value)
        elif key == "TEXTS":
            texts = int(value)
        elif key == "SAMPLE":
            samples.append(float(value))
    return load, texts, samples


def compare_embeddings(rust: Path, python: Path, n: int) -> dict[str, float]:
    import numpy as np

    a = np.fromfile(rust, dtype="<f4").reshape(n, -1)
    b = np.fromfile(python, dtype="<f4").reshape(n, -1)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    nonzero = norms > 0
    cos = (a * b).sum(axis=1)[nonzero] / norms[nonzero]
    return {
        "dim": int(a.shape[1]),
        "max_abs_diff": float(np.abs(a - b).max()),
        "min_cosine": float(cos.min()),
        "identical_rows": int((a == b).all(axis=1).sum()),
    }


def fmt_rate(x: float) -> str:
    for unit, scale in (("M", 1e6), ("K", 1e3)):
        if x >= scale:
            return f"{x / scale:.2f}{unit}"
    return f"{x:.0f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="minishlab/potion-base-8M")
    p.add_argument("--corpus", type=Path, help="text file, one text per line (default: the pinned Wikipedia sample)")
    p.add_argument("--threads", default="1,all", help="comma-separated: 'all' and/or thread counts")
    p.add_argument("--rounds", type=int, default=5, help="alternating process launches per side")
    p.add_argument("--reps", type=int, default=5, help="timed passes per process launch")
    p.add_argument("--no-python-multiprocessing", action="store_true", help="use_multiprocessing=False in Python for 'all' too")
    p.add_argument("--python-multiprocessing", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path, help="write raw results as JSON (default: target/python-compare/results-*.json)")
    p.add_argument("--python-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--embeddings", type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()

    if a.python_worker:
        python_worker(a.corpus, a.model, a.reps, a.python_multiprocessing, a.embeddings)
        return

    corpus = a.corpus.resolve() if a.corpus else ensure_corpus()
    binary = build()
    WORK.mkdir(parents=True, exist_ok=True)
    rust_base = [str(binary), f"corpus={corpus}", f"model={a.model}"]
    python_base = [sys.executable, str(Path(__file__).resolve()), "--python-worker", "--corpus", str(corpus), "--model", a.model]
    python_mp = not a.no_python_multiprocessing
    rust_cmd = [*rust_base, f"reps={a.reps}"]

    def python_cmd(threads: str) -> list[str]:
        mp = ["--python-multiprocessing"] if python_mp and threads == "all" else []
        return [*python_base, *mp, "--reps", str(a.reps)]

    rust_emb, python_emb = WORK / "rust.f32", WORK / "python.f32"
    _, n, _ = run_side([*rust_base, "reps=0", f"out={rust_emb}"], "all")
    run_side([*python_base, "--reps", "0", "--embeddings", str(python_emb)], "all")
    output = compare_embeddings(rust_emb, python_emb, n)

    results = {}
    started = time.time()
    for threads in a.threads.split(","):
        res = results[threads] = {"rust": {"load": [], "seconds": []}, "python": {"load": [], "seconds": []}}
        for r in range(a.rounds):
            print(f"threads={threads} round {r + 1}/{a.rounds}", file=sys.stderr)
            order = ("rust", "python") if r % 2 == 0 else ("python", "rust")
            for side in order:
                load, _, samples = run_side(rust_cmd if side == "rust" else python_cmd(threads), threads)
                res[side]["load"].append(load)
                res[side]["seconds"].extend(samples)

    print(f"\nmodel={a.model} corpus={corpus} texts={n} rounds={a.rounds} reps={a.reps}")
    print(f"python use_multiprocessing={python_mp} for threads=all   elapsed {time.time() - started:.0f}s\n")
    header = f"{'threads':>7} {'Rust texts/s':>14} {'Python texts/s':>15} {'speedup':>8} {'Rust load':>10} {'Python load':>12}"
    print(header)
    print("-" * len(header))
    for threads, res in results.items():
        rate = {side: statistics.median(n / s for s in res[side]["seconds"]) for side in res}
        load = {side: statistics.median(res[side]["load"]) for side in res}
        res["median_texts_per_second"], res["median_load_seconds"] = rate, load
        print(
            f"{threads:>7} {fmt_rate(rate['rust']):>14} {fmt_rate(rate['python']):>15} "
            f"{rate['rust'] / rate['python']:>7.1f}x {load['rust'] * 1e3:>8.0f}ms {load['python'] * 1e3:>10.0f}ms"
        )
    print(
        f"\nRust vs Python embeddings ({output['dim']} dims): max abs diff {output['max_abs_diff']:.2e}, "
        f"min cosine {output['min_cosine']:.7f}, {output['identical_rows']}/{n} rows bit-identical"
    )
    print("Throughput is the median over all timed passes; Python load time includes importing model2vec.")

    out = a.out or WORK / f"results-{time.strftime('%Y%m%d-%H%M%S')}.json"
    meta = {"model": a.model, "corpus": str(corpus), "texts": n, "rounds": a.rounds, "reps": a.reps, "python_multiprocessing_for_all": python_mp}
    out.write_text(json.dumps({"meta": meta, "output": output, "results": results}, indent=2))
    print(f"raw results: {out}")


if __name__ == "__main__":
    main()
