#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub", "pyarrow", "numpy", "safetensors"]
# ///
"""Rerunnable benchmark suite: model2vec-rs against itself, go-potion and encoders.

Rebuilds every input from pinned sources under `target/bench-suite/data/`:

- `big.txt` (norvig.com, checked against a SHA-256 digest): pure ASCII English.
- NanoMIRACL passages (`hakari-bench/NanoMIRACL`, pinned revision) in 18
  languages, one passage (title and text) per line, checked against digests;
  plus `en_ascii`, the English passages with every non-ASCII char removed.
- Python-style int8 copies (one scale for the whole table) of
  potion-base-8M, potion-retrieval-32M and static-retrieval-mrl-en-v1.

Sections (`--sections`, default all):

- `ab`: `scripts/bench_compare.py` on every corpus, baseline vs candidate,
  single thread, paired rounds with 95% CIs and an identical-output check.
- `int8-ab`: `scripts/bench_compare.py` on the Python-style int8 models.
- `libs`: MB/s of model2vec-rs (f32, `quantize_int8`, Python int8),
  go-potion and encoders (f32 and int8) on every corpus, one chunk per call
  on one thread (`single`) and one batch on all cores (`batch`).
- `accuracy`: cosine similarity of every library and precision to the
  model2vec-rs f32 output.

All libraries see the same texts: the corpus split on whitespace into
256-word chunks, the first 400,000 words for `single` and all of them for
`batch`. Throughput is the best of several passes per process, and the median
over `--rounds` process launches in rotating order.

go-potion is pulled from the Go module proxy at a pinned version
(`GO_POTION_VERSION`), or built from a local checkout with `--go-potion PATH`;
either needs `go`. Models the go-potion build does not support are skipped. encoders compiles one model into each binary,
so it is built once per model and precision. Either is skipped, with a notice,
when it cannot be built.

Usage (from the repository root):

    uv run scripts/bench_suite.py                           # everything
    uv run scripts/bench_suite.py --quick                   # big, en, ko; one round
    uv run scripts/bench_suite.py --sections libs --corpora big,en,ko --modes batch
    uv run scripts/bench_suite.py --sections ab --baseline v0.5.0
    uv run scripts/bench_suite.py --sections libs --revisions main,WORKTREE

Results go to `target/bench-suite/results/<timestamp>/` as `results.json`
and `report.md`; the report is also printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "target" / "bench-suite"
DATA = WORK / "data"

BIG_URL = "https://norvig.com/big.txt"
BIG_SHA256 = "fa066c7d40f0f201ac4144e652aa62430e58a6b3805ec70650f678da5804e87b"
MIRACL_REPO = "hakari-bench/NanoMIRACL"
MIRACL_REVISION = "ea97b8cca4e0e0e19910c43e7f3ee9a60fef3a74"
CORPUS_SHA256 = {
    "ar": "c39cc51fe4e0d6ca0537923ce5b694018a27262fdba9ffbea7413359c7e98c8d",
    "bn": "b6b1898068bb7b1401113bfd1209f57ded9cfb5ff9aab3c08a82e66e1388e41c",
    "de": "82766c863f9519d9d0e1afa9bfa6bc1f91ad0953549ab769dedac293579940e7",
    "en": "e0f77d47c6032e663a45822818ba664472ff0087c43c333f1f029b489570358f",
    "en_ascii": "25fd5154e3e90bdc3b9c2a0b2843ded4ebc82da05ee73a5bf82da821bda431a0",
    "es": "939677b74616037d4349a21119a30a9f3cc50d6b0577588074768cb8e3d60b26",
    "fa": "4b44e7a881b8429ceada2a8790b4bfe0b0a31b6b905d0914fcf5961e8ebddccd",
    "fi": "06479671d29c9a3ec021665f5edb0fda3fefdb890daf66d61b3ed36963d33298",
    "fr": "aed2dfbaa7330331fef343bbc7ce1aa7d31c06681aaab8f7ecfeb75a8efc8400",
    "hi": "94d90dce0c8a03cfd8e8805f3d063d82347e7a3a9ec3e658309e613639c0b313",
    "id": "36e724e833367b25c4668e4d7701522352725cdc03961293a8a81e60469c437f",
    "ja": "dc199e7de865362ce188a144cf568f844a3b5067d6ef5b52e167ef4538630b8a",
    "ko": "dd9cc78d83d6a23e6536d58e55405adce0c91ab0653ed0ce1a0d2469bb4f6c5f",
    "ru": "28d53fb3eb61c2829199752b233157140d0f796a55e453e14bc53b5b3485c02e",
    "sw": "2adfce8b1fbfd42883fd0fe3644010d46334c63d8c526e06b3f22eeb086bb3a2",
    "te": "c63ad35ea768e7075a8a42cdd655ff41537418da8f1230b647d8dbd23efbe96e",
    "th": "95bb8e304cea24f8854d5d9c971bd748820ff805e47c10fa908912e5643fab34",
    "yo": "ec2d72f8a7b69dba2ae70dcb9f7dda0b2906df8489f64dd94371f9420b15c221",
    "zh": "c49f4a1259bb96dc38c18ee28959946a4dda8cc4d6256f3e059629569abac418",
}
LANGS = ["en", "de", "es", "fr", "fi", "id", "sw", "yo", "ru", "ar", "fa", "hi", "bn", "te", "th", "zh", "ja", "ko"]
CORPORA = ["big", "en_ascii", *LANGS]

MODELS = {
    "potion-base-2M": {"hf": "minishlab/potion-base-2M", "go": "BASE2M", "enc": "potion-base-2m"},
    "potion-base-8M": {"hf": "minishlab/potion-base-8M", "go": "BASE8M", "enc": "potion-base-8m"},
    "potion-retrieval-32M": {"hf": "minishlab/potion-retrieval-32M", "go": "RETRIEVAL32M", "enc": "potion-retrieval-32m"},
    "blink-multilingual-64M": {"hf": "minishlab/blink-multilingual-64M", "go": "BLINKMULTILINGUAL64M", "enc": None},
    "blink-8m": {"hf": "minishlab/blink-8m", "go": "BLINK8M", "enc": "blink-8m"},
    "bekko-static-embedding-v1": {"hf": "hotchpotch/bekko-static-embedding-v1", "go": None, "enc": "bekko-static-embedding-v1"},
}
INT8_MODELS = ["minishlab/potion-base-8M", "minishlab/potion-retrieval-32M", "sentence-transformers/static-retrieval-mrl-en-v1"]
AB_MODELS = ["minishlab/potion-base-8M", "minishlab/blink-multilingual-64M"]
GO_POTION_MODULE = "github.com/trengrj/go-potion"
GO_POTION_VERSION = "v0.2.1-0.20261112123519-24a66aed39d7"
ENCODERS_VERSION = "=3.1.0"
ENCODERS_PARALLEL_MIN = 128
ENCODERS_MIN_LEN = 32


def encoders_all_cores(chunks: int) -> bool:
    return chunks >= max(ENCODERS_PARALLEL_MIN, ENCODERS_MIN_LEN * (os.cpu_count() or 1))
ITERS = {"single": 3, "batch": 10}
DUMP_CHUNKS = 2000

RUST_MAIN = r"""
use std::{env, fs, io::Write, time::Instant};
use model2vec_rs::model::StaticModel;

fn main() {
    let a: Vec<String> = env::args().collect();
    let (model_id, corpus, iters, mode) = (&a[1], &a[2], a[3].parse::<usize>().unwrap(), a[4].as_str());
    let int8 = a[5] == "int8";
    let text = fs::read_to_string(corpus).unwrap();
    let words: Vec<&str> = text.split_whitespace().collect();
    let words = if mode == "batch" { &words[..] } else { &words[..words.len().min(400_000)] };
    let chunks: Vec<String> = words.chunks(256).map(|c| c.join(" ")).collect();
    let bytes: usize = chunks.iter().map(|c| c.len()).sum();
    let mut model = StaticModel::from_pretrained(model_id, None, None, None).unwrap().with_max_length(None);
    if int8 {
        model = model.quantize_int8();
    }
    if mode == "dump" {
        let mut f = fs::File::create(&a[6]).unwrap();
        let n = chunks.len().min(a[7].parse().unwrap());
        for c in &chunks[..n] {
            for x in model.encode_single(c) {
                f.write_all(&x.to_le_bytes()).unwrap();
            }
        }
        println!("{n}");
        return;
    }
    if mode == "batch" {
        std::hint::black_box(model.encode(&chunks));
    }
    let mut best = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        if mode == "batch" {
            std::hint::black_box(model.encode(&chunks));
        } else {
            for c in &chunks {
                std::hint::black_box(model.encode_single(c));
            }
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("{:.3} {}", bytes as f64 / best / 1048576.0, chunks.len());
}
"""

ENCODERS_MAIN = r"""
use std::{env, fs, io::Write, time::Instant};

fn main() {
    let a: Vec<String> = env::args().collect();
    let (corpus, iters, mode) = (&a[1], a[2].parse::<usize>().unwrap(), a[3].as_str());
    let text = fs::read_to_string(corpus).unwrap();
    let words: Vec<&str> = text.split_whitespace().collect();
    let words = if mode == "batch" { &words[..] } else { &words[..words.len().min(400_000)] };
    let chunks: Vec<String> = words.chunks(256).map(|c| c.join(" ")).collect();
    let bytes: usize = chunks.iter().map(|c| c.len()).sum();
    if mode == "dump" {
        let mut f = fs::File::create(&a[4]).unwrap();
        let n = chunks.len().min(a[5].parse().unwrap());
        for c in &chunks[..n] {
            for x in encoders::encode_single(c) {
                f.write_all(&x.to_le_bytes()).unwrap();
            }
        }
        println!("{n}");
        return;
    }
    if mode == "batch" {
        std::hint::black_box(encoders::encode(&chunks));
    }
    let mut best = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        if mode == "batch" {
            std::hint::black_box(encoders::encode(&chunks));
        } else {
            for c in &chunks {
                std::hint::black_box(encoders::encode_single(c));
            }
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("{:.3}", bytes as f64 / best / 1048576.0);
}
"""

GO_MAIN = r"""
package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	potion "github.com/trengrj/go-potion"
)

func main() {
	if os.Args[1] == "models" {
		for _, m := range potion.Models() {
			fmt.Println(m)
		}
		return
	}
	model, corpus, mode := potion.Model(os.Args[1]), os.Args[2], os.Args[4]
	iters, _ := strconv.Atoi(os.Args[3])
	raw, err := os.ReadFile(corpus)
	if err != nil {
		panic(err)
	}
	words := strings.Fields(string(raw))
	if mode != "batch" && len(words) > 400000 {
		words = words[:400000]
	}
	var chunks []string
	bytes := 0
	for i := 0; i < len(words); i += 256 {
		c := strings.Join(words[i:min(i+256, len(words))], " ")
		chunks = append(chunks, c)
		bytes += len(c)
	}
	enc, err := potion.New(context.Background(), model)
	if err != nil {
		panic(err)
	}
	if mode == "dump" {
		n, _ := strconv.Atoi(os.Args[6])
		f, err := os.Create(os.Args[5])
		if err != nil {
			panic(err)
		}
		defer f.Close()
		buf := make([]byte, 4)
		n = min(n, len(chunks))
		for _, c := range chunks[:n] {
			v, err := enc.Encode(c)
			if err != nil {
				panic(err)
			}
			for _, x := range v {
				binary.LittleEndian.PutUint32(buf, math.Float32bits(x))
				f.Write(buf)
			}
		}
		fmt.Println(n)
		return
	}
	if mode == "batch" {
		if _, err := enc.EncodeMany(chunks); err != nil {
			panic(err)
		}
	}
	best := 1e18
	for it := 0; it < iters; it++ {
		t := time.Now()
		if mode == "batch" {
			if _, err := enc.EncodeMany(chunks); err != nil {
				panic(err)
			}
		} else {
			for _, c := range chunks {
				if _, err := enc.Encode(c); err != nil {
					panic(err)
				}
			}
		}
		best = min(best, time.Since(t).Seconds())
	}
	fmt.Printf("%.3f\n", float64(bytes)/best/1048576)
}
"""


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_digest(path: Path, want: str) -> None:
    got = sha256(path)
    if got != want:
        sys.exit(f"{path} has SHA-256 {got}, expected {want}; delete it to rebuild")


def corpus_path(name: str) -> Path:
    return DATA / ("big.txt" if name == "big" else f"miracl-{name}.txt")


def prepare_data() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    big = corpus_path("big")
    if not big.exists():
        log(f"downloading {BIG_URL}")
        tmp = big.with_suffix(".part")
        urllib.request.urlretrieve(BIG_URL, tmp)
        tmp.rename(big)
    check_digest(big, BIG_SHA256)

    if not all(corpus_path(lang).exists() for lang in [*LANGS, "en_ascii"]):
        import pyarrow.parquet as pq
        from huggingface_hub import snapshot_download

        log(f"downloading {MIRACL_REPO}@{MIRACL_REVISION[:12]}")
        root = Path(
            snapshot_download(MIRACL_REPO, repo_type="dataset", revision=MIRACL_REVISION, allow_patterns=["corpus/*.parquet"])
        )
        for f in sorted((root / "corpus").glob("*.parquet")):
            lang = f.name[:2]
            if lang not in LANGS:
                continue
            t = pq.read_table(f)
            cols = [c for c in t.column_names if c in ("title", "text")]
            rows = zip(*[t.column(c).to_pylist() for c in cols])
            corpus_path(lang).write_text("\n".join(" ".join(x or "" for x in r) for r in rows))
        en = corpus_path("en").read_text()
        corpus_path("en_ascii").write_text("".join(c for c in en if c.isascii()))
    for name, digest in CORPUS_SHA256.items():
        check_digest(corpus_path(name), digest)

    for repo in INT8_MODELS:
        dst = int8_model_path(repo)
        if (dst / "model.safetensors").exists():
            continue
        import numpy as np
        from huggingface_hub import snapshot_download
        from safetensors.numpy import load_file, save_file

        log(f"making Python-style int8 copy of {repo}")
        src = Path(snapshot_download(repo))
        st = next(p for p in [src / "model.safetensors", src / "0_StaticEmbedding" / "model.safetensors"] if p.exists())
        tensors = load_file(st)
        key = next(k for k in ["embeddings", "0", "embedding.weight"] if k in tensors)
        emb = tensors[key].astype(np.float32)
        tensors[key] = np.round(emb / (np.max(np.abs(emb)) / 127)).astype(np.int8)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(st.parent, dst, ignore=shutil.ignore_patterns("model.safetensors"))
        for extra in ["config.json", "config_sentence_transformers.json", "tokenizer.json"]:
            if not (dst / extra).exists() and (src / extra).exists():
                shutil.copy(src / extra, dst / extra)
        save_file(tensors, dst / "model.safetensors")


def int8_model_path(repo: str) -> Path:
    return DATA / "int8-models" / repo.replace("/", "__")


def revision_source(ref: str) -> tuple[Path, str]:
    if ref == "WORKTREE":
        return REPO, "worktree"
    commit = run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=REPO, capture_output=True).stdout.strip()
    path = WORK / f"src-{commit[:12]}"
    run(["git", "worktree", "prune"], cwd=REPO)
    if not path.exists():
        run(["git", "worktree", "add", "--detach", str(path), commit], cwd=REPO, capture_output=True)
    return path, commit[:12]


def build_rust(ref: str) -> Path:
    source, key = revision_source(ref)
    crate = WORK / "harness" / f"rust-{key}"
    (crate / "src").mkdir(parents=True, exist_ok=True)
    (crate / "src" / "main.rs").write_text(RUST_MAIN)
    (crate / "Cargo.toml").write_text(
        f'[package]\nname = "suite-rs"\nversion = "0.0.0"\nedition = "2021"\n\n'
        f"[dependencies]\nmodel2vec-rs = {{ path = {json.dumps(str(source))} }}\n\n"
        '[profile.release]\nlto = true\ncodegen-units = 1\n\n[workspace]\n'
    )
    if (source / "Cargo.lock").exists():
        shutil.copy2(source / "Cargo.lock", crate / "Cargo.lock")
    log(f"building model2vec-rs harness for {ref} ({key})")
    run(["cargo", "build", "--release", "--quiet"], cwd=crate)
    return crate / "target" / "release" / "suite-rs"


def build_go(go_potion: Path | None) -> tuple[Path, set[str]] | None:
    if shutil.which("go") is None:
        log("skipping go-potion: needs `go`")
        return None
    if go_potion is not None and not (go_potion / "go.mod").exists():
        log(f"skipping go-potion: no checkout at {go_potion}")
        return None
    mod = WORK / "harness" / "go"
    if mod.exists():
        shutil.rmtree(mod)
    mod.mkdir(parents=True)
    (mod / "main.go").write_text(GO_MAIN)
    gomod = f"module suitego\n\ngo 1.23\n\nrequire {GO_POTION_MODULE} {GO_POTION_VERSION}\n"
    if go_potion is not None:
        gomod += f"\nreplace {GO_POTION_MODULE} => {go_potion.resolve()}\n"
    (mod / "go.mod").write_text(gomod)
    log(f"building go-potion harness from {go_potion or f'{GO_POTION_MODULE}@{GO_POTION_VERSION}'}")
    try:
        run(["go", "mod", "tidy"], cwd=mod, capture_output=True)
        run(["go", "build", "-o", "suitego", "."], cwd=mod, capture_output=True)
    except subprocess.CalledProcessError as e:
        log(f"skipping go-potion: build failed\n{e.stderr}")
        return None
    binary = mod / "suitego"
    return binary, set(run([str(binary), "models"], capture_output=True).stdout.split())


def build_encoders(models: list[str]) -> dict[tuple[str, str], Path]:
    crate = WORK / "harness" / "encoders"
    (crate / "src").mkdir(parents=True, exist_ok=True)
    (crate / "src" / "main.rs").write_text(ENCODERS_MAIN)
    features = sorted({MODELS[m]["enc"] for m in models if MODELS[m]["enc"]} | {"float32", "int8"})
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "suite-enc"\nversion = "0.0.0"\nedition = "2024"\n\n'
        f'[dependencies]\nencoders = {{ version = "{ENCODERS_VERSION}", default-features = false, features = ["std", "rayon"] }}\n\n'
        "[features]\n" + "".join(f'{f} = ["encoders/{f}"]\n' for f in features) + "\n"
        '[profile.release]\nlto = "fat"\ncodegen-units = 1\n\n[workspace]\n'
    )
    bins = {}
    for model in models:
        enc = MODELS[model]["enc"]
        if enc is None:
            continue
        for prec in ["float32", "int8"]:
            out = crate / "bin" / f"{enc}-{prec}"
            if not out.exists():
                log(f"building encoders harness for {enc} {prec}")
                target = crate / "target" / f"{enc}-{prec}"
                try:
                    run(
                        ["cargo", "build", "--release", "--quiet", "--features", f"{enc},{prec}"],
                        cwd=crate,
                        env=dict(os.environ, CARGO_TARGET_DIR=str(target)),
                        capture_output=True,
                    )
                except subprocess.CalledProcessError as e:
                    log(f"skipping encoders: build failed\n{e.stderr}")
                    return {}
                out.parent.mkdir(exist_ok=True)
                shutil.copy2(target / "release" / "suite-enc", out)
            bins[(model, "f32" if prec == "float32" else "int8")] = out
    return bins


def entrants(model: str, rust: dict[str, Path], go: tuple[Path, set[str]] | None, enc: dict) -> list[tuple[str, list[str]]]:
    hf = MODELS[model]["hf"]
    out = []
    for ref, binary in rust.items():
        out.append((f"rs:{ref} f32", [str(binary), hf, "{corpus}", "{iters}", "{mode}", "f32"]))
        out.append((f"rs:{ref} int8", [str(binary), hf, "{corpus}", "{iters}", "{mode}", "int8"]))
        if hf in INT8_MODELS:
            out.append((f"rs:{ref} py-int8", [str(binary), str(int8_model_path(hf)), "{corpus}", "{iters}", "{mode}", "f32"]))
    if go is not None and MODELS[model]["go"] in go[1]:
        out.append(("go-potion f32", [str(go[0]), MODELS[model]["go"], "{corpus}", "{iters}", "{mode}"]))
    for prec in ["f32", "int8"]:
        if (model, prec) in enc:
            out.append((f"encoders {prec}", [str(enc[(model, prec)]), "{corpus}", "{iters}", "{mode}"]))
    return out


def thread_env(mode: str) -> dict[str, str]:
    env = dict(os.environ)
    if mode == "single":
        env.update(RAYON_NUM_THREADS="1", RAYON_RS_NUM_THREADS="1", GOMAXPROCS="1", TOKENIZERS_PARALLELISM="false")
    return env


def section_libs(a, rust, go, enc) -> list[dict]:
    rows = []
    configs = [(c, m, mode) for c in a.corpora for m in a.models for mode in a.modes]
    for i, (corpus, model, mode) in enumerate(configs, 1):
        log(f"[libs {i}/{len(configs)}] {corpus} {model} {mode}")
        ents = entrants(model, rust, go, enc)
        rates: dict[str, list[float]] = {name: [] for name, _ in ents}
        chunks = 0
        for r in range(a.rounds):
            order = ents[r % len(ents) :] + ents[: r % len(ents)]
            for name, cmd in order:
                args = [x.format(corpus=corpus_path(corpus), iters=ITERS[mode], mode=mode) for x in cmd]
                out = run(args, env=thread_env(mode), capture_output=True).stdout.split()
                rates[name].append(float(out[0]))
                if name.startswith("rs:"):
                    chunks = int(out[1])
        rows.append(
            {
                "corpus": corpus,
                "model": model,
                "mode": mode,
                "chunks": chunks,
                "mb_per_s": {k: statistics.median(v) for k, v in rates.items()},
                "rounds": rates,
            }
        )
    return rows


def section_accuracy(a, rust, go, enc) -> list[dict]:
    import numpy as np

    dumps = WORK / "dumps"
    dumps.mkdir(parents=True, exist_ok=True)
    rows = []
    for corpus in a.accuracy_corpora:
        for model in a.models:
            log(f"[accuracy] {corpus} {model}")
            vecs = {}
            for name, cmd in entrants(model, rust, go, enc):
                path = dumps / f"{corpus}-{model}-{name.replace(' ', '_').replace(':', '_')}.bin"
                args = [x.format(corpus=corpus_path(corpus), iters=1, mode="dump") for x in cmd] + [str(path), str(DUMP_CHUNKS)]
                chunks = int(run(args, capture_output=True).stdout.split()[-1])
                vecs[name] = np.fromfile(path, dtype="<f4").reshape(chunks, -1)
            ref_name = f"rs:{a.revisions[-1]} f32"
            ref = vecs[ref_name]
            for name, v in vecs.items():
                if name == ref_name:
                    continue
                n = min(len(v), len(ref))
                x, y = v[:n].astype(np.float64), ref[:n].astype(np.float64)
                cos = (x * y).sum(1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
                rows.append(
                    {
                        "corpus": corpus,
                        "model": model,
                        "variant": name,
                        "reference": ref_name,
                        "chunks": int(n),
                        "mean": float(cos.mean()),
                        "p1": float(np.percentile(cos, 1)),
                        "min": float(cos.min()),
                        "bit_identical": bool(np.array_equal(v[:n], ref[:n])),
                    }
                )
    return rows


def section_ab(a, models: list[str], corpora: list[str], threads: str, label: str, out_dir: Path) -> list[dict]:
    rows = []
    for corpus in corpora:
        log(f"[{label}] {corpus}")
        out = out_dir / f"{label}-{corpus}.json"
        cmd = [
            sys.executable,
            str(REPO / "scripts" / "bench_compare.py"),
            "--baseline", a.baseline,
            "--candidate", a.candidate,
            "--models", ",".join(models),
            "--modes", "batch",
            "--threads", threads,
            "--rounds", str(a.ab_rounds),
            "--corpus", str(corpus_path(corpus)),
            "--out", str(out),
        ]
        run(cmd, stdout=subprocess.DEVNULL)
        for r in json.loads(out.read_text())["results"]:
            rows.append(
                {
                    "corpus": corpus,
                    "model": Path(r["model"]).name,
                    "workload": r["workload"],
                    "threads": r["threads"],
                    "speedup": r["speedup"],
                    "ci95": r["ci95"],
                    "identical_output": r["identical_output"],
                }
            )
    return rows


def fmt_ab(rows: list[dict], title: str, a) -> list[str]:
    lines = [f"## {title}", "", f"Speedup of {a.candidate} over {a.baseline} (median of paired rounds, 95% CI).", ""]
    lines += ["| corpus | model | workload | threads | speedup | 95% CI | output |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lo, hi = r["ci95"]
        flag = " faster" if lo > 1 else " SLOWER" if hi < 1 else ""
        lines.append(
            f"| {r['corpus']} | {r['model']} | {r['workload']} | {r['threads']} | {r['speedup']:.3f}x{flag} "
            f"| [{lo:.3f}, {hi:.3f}] | {'identical' if r['identical_output'] else 'DIFFERENT'} |"
        )
    return lines + [""]


def fmt_libs(rows: list[dict], a) -> list[str]:
    lines = ["## Libraries (MB/s)", "", "`single`: one chunk per call, one thread. `batch`: one call, all cores.", ""]
    ref = a.revisions[-1]
    for mode in a.modes:
        sub = [r for r in rows if r["mode"] == mode]
        if not sub:
            continue
        names = list(dict.fromkeys(n for r in sub for n in r["mb_per_s"]))
        ratios = [("rs/go", f"rs:{ref} f32", "go-potion f32"), ("rs/enc f32", f"rs:{ref} f32", "encoders f32"), ("rs/enc int8", f"rs:{ref} int8", "encoders int8")]
        ratios = [r for r in ratios if r[2] in names]
        lines += [f"### {mode}", "", "| corpus | model | " + " | ".join(names) + " | " + " | ".join(r[0] for r in ratios) + " |"]
        lines.append("|---|---|" + "---|" * (len(names) + len(ratios)))
        for r in sub:
            v = r["mb_per_s"]
            cells = [f"{v[n]:.1f}" if n in v else "-" for n in names]
            cells += [f"{v[x] / v[y]:.2f}" if x in v and y in v else "-" for _, x, y in ratios]
            serial = mode == "batch" and not encoders_all_cores(r["chunks"]) and any(n.startswith("encoders") for n in v)
            lines.append(f"| {r['corpus']}{' †' if serial else ''} | {r['model']} | " + " | ".join(cells) + " |")
        lines.append("")
    if any(r["mode"] == "batch" and not encoders_all_cores(r["chunks"]) for r in rows):
        lines += [
            f"† too few chunks for encoders to use every core: it runs batches of fewer than {ENCODERS_PARALLEL_MIN} texts "
            f"on one thread and splits larger ones into groups of at least {ENCODERS_MIN_LEN}, so batches under "
            f"{ENCODERS_MIN_LEN} × {os.cpu_count()} cores texts are not all-core numbers.",
            "",
        ]
    return lines


def fmt_accuracy(rows: list[dict]) -> list[str]:
    lines = ["## Accuracy", "", f"Cosine similarity to the model2vec-rs f32 output, first {DUMP_CHUNKS} chunks.", ""]
    lines += ["| corpus | model | variant | mean | p1 | min | bit-identical |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['corpus']} | {r['model']} | {r['variant']} | {r['mean']:.6f} | {r['p1']:.6f} | {r['min']:.6f} | {'yes' if r['bit_identical'] else 'no'} |"
        )
    return lines + [""]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sections", default="ab,int8-ab,libs,accuracy", help="comma-separated: ab, int8-ab, libs, accuracy, data")
    p.add_argument("--corpora", default=",".join(CORPORA), help=f"comma-separated subset of {','.join(CORPORA)}")
    p.add_argument("--accuracy-corpora", default="big,en,ko,zh,ru")
    p.add_argument(
        "--models",
        help=f"comma-separated subset of {','.join(MODELS)} (default: all for `libs` and `accuracy`, "
        f"{','.join(Path(m).name for m in AB_MODELS)} for `ab`)",
    )
    p.add_argument("--modes", default="single,batch")
    p.add_argument("--rounds", type=int, default=2, help="process launches per library in `libs`")
    p.add_argument("--revisions", default="WORKTREE", help="model2vec-rs revisions for `libs` and `accuracy`; the last is the reference")
    p.add_argument("--baseline", default="main", help="baseline revision for `ab` and `int8-ab`")
    p.add_argument("--candidate", default="WORKTREE", help="candidate revision for `ab` and `int8-ab`")
    p.add_argument("--ab-rounds", type=int, default=8)
    p.add_argument("--go-potion", type=Path, help=f"local go-potion checkout to use instead of {GO_POTION_MODULE}@{GO_POTION_VERSION}")
    p.add_argument("--no-go", action="store_true")
    p.add_argument("--no-encoders", action="store_true")
    p.add_argument("--quick", action="store_true", help="corpora big,en,ko; one round; 3 A/B rounds")
    a = p.parse_args()
    if a.quick:
        a.corpora, a.accuracy_corpora, a.rounds, a.ab_rounds = "big,en,ko", "en,ko", 1, 3
    sections = set(a.sections.split(","))
    a.corpora = a.corpora.split(",")
    a.accuracy_corpora = a.accuracy_corpora.split(",")
    a.models = a.models.split(",") if a.models else None
    a.modes = a.modes.split(",")
    a.revisions = a.revisions.split(",")
    for c in a.corpora + a.accuracy_corpora:
        if c not in CORPORA:
            sys.exit(f"unknown corpus {c}")
    for m in a.models or []:
        if m not in MODELS:
            sys.exit(f"unknown model {m}")
    ab_models = [MODELS[m]["hf"] for m in a.models] if a.models else AB_MODELS
    a.models = a.models or list(MODELS)

    started = time.time()
    prepare_data()
    out_dir = WORK / "results" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "meta": {
            "sections": sorted(sections),
            "revisions": a.revisions,
            "baseline": a.baseline,
            "candidate": a.candidate,
            "rounds": a.rounds,
            "ab_rounds": a.ab_rounds,
            "commit": run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True).stdout.strip(),
            "dirty": bool(run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True).stdout.strip()),
        }
    }
    report = ["# model2vec-rs benchmark suite", "", f"commit {results['meta']['commit'][:12]}{' (dirty)' if results['meta']['dirty'] else ''}", ""]

    if "ab" in sections:
        results["ab"] = section_ab(a, ab_models, a.corpora, "1", "ab", out_dir)
        report += fmt_ab(results["ab"], "A/B per corpus (single thread)", a)
    if "int8-ab" in sections:
        results["int8_ab"] = section_ab(a, [str(int8_model_path(m)) for m in INT8_MODELS], ["big"], "1,all", "int8-ab", out_dir)
        report += fmt_ab(results["int8_ab"], "A/B on Python-style int8 models (big.txt)", a)
    if sections & {"libs", "accuracy"}:
        rust = {ref: build_rust(ref) for ref in a.revisions}
        go = None if a.no_go else build_go(a.go_potion)
        enc = {} if a.no_encoders else build_encoders(a.models)
        go_source = str(a.go_potion.resolve()) if a.go_potion else f"{GO_POTION_MODULE}@{GO_POTION_VERSION}"
        results["libraries"] = {"go-potion": go_source if go else None, "encoders": ENCODERS_VERSION if enc else None}
        if go:
            report += [f"go-potion: {go_source}", ""]
            skipped = [m for m in a.models if MODELS[m]["go"] not in go[1]]
            if skipped:
                report += [f"go-potion does not support {', '.join(skipped)}; skipped there.", ""]
        if "libs" in sections:
            results["libs"] = section_libs(a, rust, go, enc)
            report += fmt_libs(results["libs"], a)
        if "accuracy" in sections:
            results["accuracy"] = section_accuracy(a, rust, go, enc)
            report += fmt_accuracy(results["accuracy"])

    report.append(f"elapsed {time.time() - started:.0f}s")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    log(f"results: {out_dir}")


if __name__ == "__main__":
    main()
