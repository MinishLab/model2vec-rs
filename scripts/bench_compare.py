#!/usr/bin/env python3
"""A/B throughput comparison of two revisions of model2vec-rs.

Builds `benches/throughput.rs` (copied from this checkout, so both sides run
the identical benchmark) against a baseline and a candidate revision, then
runs them in alternating order (A B, B A, A B, ...) so slow drift such as
thermal throttling or background load hits both sides equally. For every
configuration it reports:

- median throughput (texts/s) of each side and the interquartile range,
- the speedup (candidate / baseline) as the median of per-round paired
  ratios, with a 95% bootstrap confidence interval over rounds,
- whether both sides produced bit-identical embeddings.

Usage (from the repository root):

    python3 scripts/bench_compare.py                        # main vs working tree
    python3 scripts/bench_compare.py --baseline main --candidate main   # A/A noise check
    python3 scripts/bench_compare.py --threads 1 --models minishlab/potion-base-8M
    python3 scripts/bench_compare.py --quick                # fast smoke run

A revision is any git ref, WORKTREE for the current checkout including
uncommitted changes, or a path to a source directory. Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "target" / "bench-compare"
CORPUS_URL = "https://norvig.com/big.txt"
DEFAULT_MODELS = [
    "minishlab/potion-base-8M",
    "minishlab/potion-retrieval-32M",
    "sentence-transformers/static-retrieval-mrl-en-v1",
    "minishlab/potion-multilingual-128M",
    str(REPO / "tests" / "fixtures" / "test-model-vocab-quantized"),
]
MIN_ROUNDS_FOR_CI = 5
BENCH_ENTRY = '\n[[bench]]\nname = "throughput"\nharness = false\n'


@dataclass
class Side:
    label: str
    ref: str
    source: Path
    build_key: str
    binary: Path | None = None


@dataclass
class Result:
    samples: dict[str, list[float]] = field(default_factory=lambda: {"A": [], "B": []})
    rounds: list[tuple[float, float]] = field(default_factory=list)
    checksums: dict[str, set[str]] = field(default_factory=lambda: {"A": set(), "B": set()})
    texts: int = 0
    dim: int = 0


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def ensure_corpus() -> Path:
    path = WORK / "big.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading corpus {CORPUS_URL}", file=sys.stderr)
        with urllib.request.urlopen(CORPUS_URL) as resp:
            path.write_bytes(resp.read())
    return path


def prepare_source(label: str, ref: str) -> tuple[Path, str]:
    if ref == "WORKTREE":
        return REPO, "worktree"
    if Path(ref).is_dir():
        path = Path(ref).resolve()
        (path / "benches").mkdir(exist_ok=True)
        shutil.copy2(REPO / "benches" / "throughput.rs", path / "benches" / "throughput.rs")
        cargo_toml = path / "Cargo.toml"
        manifest = cargo_toml.read_text()
        if not re.search(r'name\s*=\s*"throughput"', manifest):
            cargo_toml.write_text(manifest + BENCH_ENTRY)
        digest = hashlib.sha256(str(path).encode()).hexdigest()[:12]
        return path, f"dir-{re.sub(r'[^A-Za-z0-9_.-]', '_', path.name)}-{digest}"
    commit = run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=REPO, capture_output=True).stdout.strip()
    path = WORK / f"src-{label}"
    run(["git", "worktree", "prune"], cwd=REPO)
    if path.exists():
        current = run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True).stdout.strip()
        if current != commit:
            run(["git", "worktree", "remove", "--force", str(path)], cwd=REPO)
    if not path.exists():
        run(["git", "worktree", "add", "--detach", str(path), commit], cwd=REPO, capture_output=True)
    (path / "benches").mkdir(exist_ok=True)
    shutil.copy2(REPO / "benches" / "throughput.rs", path / "benches" / "throughput.rs")
    cargo_toml = path / "Cargo.toml"
    manifest = cargo_toml.read_text()
    if not re.search(r'name\s*=\s*"throughput"', manifest):
        cargo_toml.write_text(manifest + BENCH_ENTRY)
    return path, commit[:12]


def build(side: Side, native: bool) -> Path:
    env = dict(os.environ, CARGO_TARGET_DIR=str(WORK / f"target-{side.build_key}"))
    if native:
        env["RUSTFLAGS"] = (env.get("RUSTFLAGS", "") + " -Ctarget-cpu=native").strip()
    print(f"building {side.label} ({side.ref}) from {side.source}", file=sys.stderr)
    out = run(
        ["cargo", "bench", "--bench", "throughput", "--no-run", "--message-format=json"],
        cwd=side.source,
        env=env,
        capture_output=True,
    ).stdout
    for line in out.splitlines():
        msg = json.loads(line)
        if msg.get("reason") == "compiler-artifact" and msg["target"]["name"] == "throughput" and msg.get("executable"):
            return Path(msg["executable"])
    raise RuntimeError(f"could not find throughput executable for {side.label}")


def run_bench(binary: Path, args: dict[str, str], threads: str) -> tuple[list[float], str, int, int]:
    env = dict(os.environ)
    if threads != "all":
        env["RAYON_NUM_THREADS"] = threads
        env["RAYON_RS_NUM_THREADS"] = threads
        if threads == "1":
            env["TOKENIZERS_PARALLELISM"] = "false"
    cmd = [str(binary), *(f"{k}={v}" for k, v in args.items())]
    out = run(cmd, env=env, capture_output=True).stdout
    samples, checksum, texts, dim = [], "", 0, 0
    for line in out.splitlines():
        if line.startswith("SAMPLE "):
            samples.append(float(line.split()[1]))
        elif line.startswith("CHECKSUM "):
            checksum = line.split()[1]
        elif line.startswith("CONFIG "):
            fields = dict(kv.split("=", 1) for kv in line.split()[1:])
            texts, dim = int(fields["texts"]), int(fields["dim"])
    return samples, checksum, texts, dim


def bootstrap_ci(ratios: list[float], iters: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(iters))
    return meds[int(0.025 * iters)], meds[int(0.975 * iters) - 1]


def quartiles(values: list[float]) -> tuple[float, float, float]:
    if len(values) < 2:
        return values[0], values[0], values[0]
    q1, q2, q3 = statistics.quantiles(values, n=4)
    return q1, q2, q3


def fmt_rate(x: float) -> str:
    for unit, scale in (("M", 1e6), ("K", 1e3)):
        if x >= scale:
            return f"{x / scale:.2f}{unit}"
    return f"{x:.0f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", default="main", help="git ref or WORKTREE (default: main)")
    p.add_argument("--candidate", default="WORKTREE", help="git ref or WORKTREE (default: WORKTREE)")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma-separated model ids or paths")
    p.add_argument("--workloads", default="queries,docs")
    p.add_argument("--modes", default="batch,single")
    p.add_argument("--threads", default="all,1", help="comma-separated: 'all' and/or thread counts")
    p.add_argument("--rounds", type=int, default=10, help="alternating process launches per side")
    p.add_argument("--reps", type=int, default=5, help="timed passes per process launch")
    p.add_argument("--native", action="store_true", help="build with -Ctarget-cpu=native")
    p.add_argument("--quick", action="store_true", help="3 rounds x 3 reps, first model only")
    p.add_argument("--out", type=Path, help="write raw results as JSON (default: target/bench-compare/results-*.json)")
    a = p.parse_args()
    if a.quick:
        a.rounds, a.reps, a.models = 3, 3, a.models.split(",")[0]

    corpus = ensure_corpus()
    sides = {
        "A": Side("A", a.baseline, *prepare_source("A", a.baseline)),
        "B": Side("B", a.candidate, *prepare_source("B", a.candidate)),
    }
    for side in sides.values():
        side.binary = build(side, a.native)
    digests = {k: hashlib.sha256(s.binary.read_bytes()).hexdigest() for k, s in sides.items()}
    same_source = sides["A"].build_key == sides["B"].build_key
    if digests["A"] == digests["B"] and not same_source:
        sys.exit(f"baseline and candidate built to identical binaries ({sides['A'].binary}); refusing to compare")

    models = [m for m in a.models.split(",") if m]
    configs = [
        (model, workload, mode, threads)
        for model in models
        for workload in a.workloads.split(",")
        for mode in a.modes.split(",")
        for threads in a.threads.split(",")
    ]
    for model in models:
        for side in sides.values():
            run_bench(side.binary, {"corpus": str(corpus), "model": model, "workload": "docs", "mode": "single", "reps": "0"}, "all")

    results: dict[tuple, Result] = {}
    started = time.time()
    for i, cfg in enumerate(configs, 1):
        model, workload, mode, threads = cfg
        res = results[cfg] = Result()
        args = {"corpus": str(corpus), "model": model, "workload": workload, "mode": mode, "reps": str(a.reps)}
        print(f"[{i}/{len(configs)}] {Path(model).name} {workload} {mode} threads={threads}", file=sys.stderr)
        for r in range(a.rounds):
            order = ("A", "B") if r % 2 == 0 else ("B", "A")
            medians = {}
            for key in order:
                samples, checksum, res.texts, res.dim = run_bench(sides[key].binary, args, threads)
                res.samples[key].extend(samples)
                res.checksums[key].add(checksum)
                medians[key] = statistics.median(samples)
            res.rounds.append((medians["A"], medians["B"]))

    print(f"\nA = {a.baseline}   B = {a.candidate}   rounds={a.rounds} reps={a.reps} native={a.native}")
    print(f"elapsed {time.time() - started:.0f}s\n")
    header = f"{'model':<34} {'workload':<8} {'mode':<6} {'thr':>3} {'dim':>5} {'A texts/s':>16} {'B texts/s':>16} {'speedup':>8} {'95% CI':>15} {'output':>9}"
    print(header)
    print("-" * len(header))
    summary = []
    for (model, workload, mode, threads), res in results.items():
        rate = {k: [res.texts / s for s in v] for k, v in res.samples.items()}
        qa, qb = quartiles(rate["A"]), quartiles(rate["B"])
        ratios = [ta / tb for ta, tb in res.rounds]
        speedup = statistics.median(ratios)
        lo, hi = bootstrap_ci(ratios)
        same = len(res.checksums["A"]) == 1 and res.checksums["A"] == res.checksums["B"]
        flag = ""
        if a.rounds >= MIN_ROUNDS_FOR_CI and not lo <= 1.0 <= hi:
            flag = "  faster" if lo > 1.0 else "  SLOWER"
        print(
            f"{Path(model).name[:34]:<34} {workload:<8} {mode:<6} {threads:>3} {res.dim:>5} "
            f"{fmt_rate(qa[1]):>7} ±{fmt_rate((qa[2] - qa[0]) / 2):>7} "
            f"{fmt_rate(qb[1]):>7} ±{fmt_rate((qb[2] - qb[0]) / 2):>7} "
            f"{speedup:>7.3f}x [{lo:.3f}, {hi:.3f}] {'identical' if same else 'DIFFERENT':>9}{flag}"
        )
        summary.append(
            {
                "model": model,
                "workload": workload,
                "mode": mode,
                "threads": threads,
                "dim": res.dim,
                "texts_per_pass": res.texts,
                "seconds": res.samples,
                "round_medians": res.rounds,
                "speedup": speedup,
                "ci95": [lo, hi],
                "checksums": {k: sorted(v) for k, v in res.checksums.items()},
                "identical_output": same,
            }
        )
    print("\n± is half the interquartile range. speedup = B/A throughput, median of paired per-round ratios.")
    if a.rounds >= MIN_ROUNDS_FOR_CI:
        print("'faster'/'SLOWER' marks configurations whose 95% CI excludes 1.0.")
    else:
        print(f"fewer than {MIN_ROUNDS_FOR_CI} rounds: confidence intervals are not meaningful, nothing is flagged.")

    out = a.out or WORK / f"results-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"baseline": a.baseline, "candidate": a.candidate, "rounds": a.rounds, "reps": a.reps, "native": a.native}
    out.write_text(json.dumps({"meta": meta, "results": summary}, indent=2))
    print(f"raw results: {out}")


if __name__ == "__main__":
    main()
