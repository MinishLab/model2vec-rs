#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub", "pyarrow"]
# ///
"""NanoBEIR retrieval-quality gate for model2vec-rs.

Downloads the 13 NanoBEIR datasets, builds `benches/nanobeir.rs` from this
checkout, scores every configured model (NDCG@10 per dataset), and compares
the scores with `benches/nanobeir_baseline.json`. Exits with status 1 if any
model's mean score drops by more than `--tolerance`, or any single dataset
score drops by more than `--dataset-tolerance`.

Usage (from the repository root):

    uv run scripts/nanobeir.py                              # gate against the baseline
    uv run scripts/nanobeir.py --models minishlab/potion-base-8M
    uv run scripts/nanobeir.py --update-baseline            # record new scores

A model is a Hugging Face id or local path, optionally suffixed with `:int8`
to score it after `StaticModel::quantize_int8`, or with `:model2vec-int8` to
score it after quantizing it to int8 with Python model2vec (saved under
`target/nanobeir/models/`). By default every model in the baseline file is
scored.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "target" / "nanobeir"
BASELINE = REPO / "benches" / "nanobeir_baseline.json"
DATASETS = [
    "NanoArguAna",
    "NanoClimateFEVER",
    "NanoDBPedia",
    "NanoFEVER",
    "NanoFiQA2018",
    "NanoHotpotQA",
    "NanoMSMARCO",
    "NanoNFCorpus",
    "NanoNQ",
    "NanoQuoraRetrieval",
    "NanoSCIDOCS",
    "NanoSciFact",
    "NanoTouche2020",
]
SPLITS = ["corpus", "queries", "qrels"]
MODEL2VEC = "model2vec==0.10.0"
QUANTIZE_SCRIPT = """
import sys
from model2vec import StaticModel
StaticModel.from_pretrained(sys.argv[1], quantize_to="int8").save_pretrained(sys.argv[2])
"""


def ensure_data() -> Path:
    data = WORK / "data"
    missing = [name for name in DATASETS if not all((data / name / f"{s}.jsonl").exists() for s in SPLITS)]
    if not missing:
        return data
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    for name in missing:
        repo = f"zeta-alpha-ai/{name}"
        print(f"downloading {repo}", file=sys.stderr)
        files = list_repo_files(repo, repo_type="dataset")
        out = data / name
        out.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            parts = sorted(f for f in files if f.startswith(f"{split}/") and f.endswith(".parquet"))
            if not parts:
                sys.exit(f"{repo} has no parquet files for split {split!r}")
            tmp = out / f"{split}.jsonl.tmp"
            with tmp.open("w") as fh:
                for part in parts:
                    for row in pq.read_table(hf_hub_download(repo, part, repo_type="dataset")).to_pylist():
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            tmp.rename(out / f"{split}.jsonl")
    return data


def model2vec_int8(model: str) -> Path:
    out = WORK / "models" / f"{model.replace('/', '--')}-int8"
    if not (out / "model.safetensors").exists():
        print(f"quantizing {model} to int8 with {MODEL2VEC}", file=sys.stderr)
        tmp = out.with_name(out.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(
            ["uv", "run", "--no-project", "--with", MODEL2VEC, "python", "-I", "-c", QUANTIZE_SCRIPT, model, str(tmp)],
            check=True,
        )
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
    return out


def build() -> Path:
    print("building benches/nanobeir.rs", file=sys.stderr)
    out = subprocess.run(
        ["cargo", "bench", "--bench", "nanobeir", "--no-run", "--message-format=json"],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    for line in out.splitlines():
        msg = json.loads(line)
        if msg.get("reason") == "compiler-artifact" and msg["target"]["name"] == "nanobeir" and msg.get("executable"):
            return Path(msg["executable"])
    raise RuntimeError("could not find nanobeir executable")


def score(binary: Path, data: Path, spec: str) -> dict[str, float]:
    model, _, quantize = spec.partition(":")
    if quantize == "model2vec-int8":
        model, quantize = str(model2vec_int8(model)), "none"
    cmd = [str(binary), f"data={data}", f"model={model}", f"quantize={quantize or 'none'}"]
    out = subprocess.run(cmd, check=True, text=True, capture_output=True).stdout
    scores = {}
    for line in out.splitlines():
        if line.startswith("SCORE "):
            _, name, value = line.split()
            scores[name] = float(value)
    if sorted(scores) != DATASETS:
        sys.exit(f"{spec}: expected scores for {DATASETS}, got {sorted(scores)}")
    return scores


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", help="comma-separated models (default: every model in the baseline)")
    p.add_argument("--baseline", type=Path, default=BASELINE)
    p.add_argument("--tolerance", type=float, default=0.002, help="max allowed drop in mean NDCG@10")
    p.add_argument("--dataset-tolerance", type=float, default=0.01, help="max allowed drop in any dataset's NDCG@10")
    p.add_argument("--update-baseline", action="store_true", help="write the scores to the baseline file")
    p.add_argument("--out", type=Path, help="write scores as JSON (default: target/nanobeir/results-*.json)")
    a = p.parse_args()

    baseline = json.loads(a.baseline.read_text()) if a.baseline.exists() else {}
    models = [m for m in a.models.split(",") if m] if a.models else list(baseline)
    if not models:
        sys.exit("no models given and the baseline is empty; pass --models")

    data = ensure_data()
    binary = build()
    results: dict[str, dict[str, float]] = {}
    for spec in models:
        started = time.time()
        results[spec] = score(binary, data, spec)
        print(f"scored {spec} in {time.time() - started:.0f}s", file=sys.stderr)

    failures = []
    for spec, scores in results.items():
        mean = statistics.mean(scores.values())
        base = baseline.get(spec)
        print(f"\n{spec}")
        print(f"  {'dataset':<20} {'NDCG@10':>8} {'baseline':>9} {'delta':>8}")
        for name in DATASETS:
            if base is None:
                print(f"  {name:<20} {scores[name]:>8.4f}")
                continue
            delta = scores[name] - base[name]
            flag = "  REGRESSION" if delta < -a.dataset_tolerance else ""
            if flag:
                failures.append(f"{spec} {name}: {base[name]:.4f} -> {scores[name]:.4f}")
            print(f"  {name:<20} {scores[name]:>8.4f} {base[name]:>9.4f} {delta:>+8.4f}{flag}")
        if base is None:
            print(f"  {'mean':<20} {mean:>8.4f}   (no baseline)")
            continue
        base_mean = statistics.mean(base[name] for name in DATASETS)
        delta = mean - base_mean
        flag = "  REGRESSION" if delta < -a.tolerance else ""
        if flag:
            failures.append(f"{spec} mean: {base_mean:.4f} -> {mean:.4f}")
        print(f"  {'mean':<20} {mean:>8.4f} {base_mean:>9.4f} {delta:>+8.4f}{flag}")

    out = a.out or WORK / f"results-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nscores: {out}")

    if a.update_baseline:
        baseline.update({spec: {name: round(s[name], 6) for name in DATASETS} for spec, s in results.items()})
        a.baseline.write_text(json.dumps(baseline, indent=2) + "\n")
        print(f"baseline updated: {a.baseline}")
        return
    if failures:
        print(f"\nFAILED: {len(failures)} regression(s) beyond tolerance:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("\nPASSED")


if __name__ == "__main__":
    main()
