# Benchmarks

- `nanobeir.rs`: retrieval quality (NDCG@10) of one model on NanoBEIR.
  Normally run through `scripts/nanobeir.py`, not directly.
- `encode.rs`: latency percentiles of `StaticModel::encode` for one revision.
- `quantize.rs`: throughput and accuracy of `StaticModel::quantize_int8`
  against the f32 model, in one process (see below).
- `throughput.rs`: throughput of one (model, workload, mode) configuration.
  Uses only the public API, so it builds against any revision. Normally run
  through `scripts/bench_compare.py`, not directly.
- `python_compare.rs`: Rust side of the comparison with Python model2vec
  in the main README. Normally run through `scripts/python_compare.py`.

## Comparing two revisions

```sh
python3 scripts/bench_compare.py                      # main vs the working tree
python3 scripts/bench_compare.py --baseline v0.3.0 --candidate my-branch
```

The script checks the baseline out into a git worktree under
`target/bench-compare/`, copies this `throughput.rs` into it, builds each
commit (and the working tree) into its own target directory, refuses to run
if both sides produce the same binary, and runs them alternately
(A B, B A, ...) so drift affects both equally. Per configuration it reports
median throughput, the paired speedup with a 95% bootstrap confidence
interval, and whether both sides produced bit-identical embeddings. Raw
samples are written to `target/bench-compare/results-*.json`.

Configurations are the product of `--models`, `--workloads` (`queries`:
8-word texts, `docs`: 256-word texts from `big.txt`), `--modes` (`batch`:
`encode`, `single`: `encode_single` in a loop) and `--threads` (`all`, or a
count; `1` also disables tokenizer parallelism). The full default run takes
about 35 minutes, most of it on the two Unigram models, whose tokenization
is far slower than pooling; pass `--models` to restrict it.

For trustworthy numbers:

1. Run an A/A comparison first (`--baseline main --candidate main`). Its
   speedups show the noise floor of the machine; differences of that size
   in a real comparison mean nothing.
2. Use the default 10 rounds or more; fewer than 5 rounds are not flagged.
3. Keep the machine idle and on power. `--threads 1` is the least noisy
   setting and isolates per-text cost from scheduling.
4. Only claim a speedup for configurations marked `faster` whose output is
   `identical` (for changes that must not alter embeddings).

## Full suite: corpora, languages and other libraries

```sh
uv run scripts/bench_suite.py --quick                 # smoke run: big, en, ko
uv run scripts/bench_suite.py                         # everything
uv run scripts/bench_suite.py --sections libs --corpora en,ko --modes batch
```

`scripts/bench_suite.py` rebuilds its inputs from pinned sources under
`target/bench-suite/data/`, checking each corpus against a SHA-256 digest:
`big.txt`, NanoMIRACL passages in 18 languages (`hakari-bench/NanoMIRACL`,
pinned revision), the English passages with non-ASCII chars removed, and
Python-style int8 copies (one scale for the whole table) of three models.
It then runs:

- `ab`: `bench_compare.py` on every corpus, `--baseline` vs `--candidate`.
- `int8-ab`: `bench_compare.py` on the int8 copies.
- `libs`: MB/s of model2vec-rs (f32, `quantize_int8`, Python int8),
  [go-potion](https://github.com/trengrj/go-potion) at a pinned version
  from the Go module proxy (or a local checkout with `--go-potion PATH`), and
  [encoders](https://crates.io/crates/encoders) (f32 and int8), on 256-word
  chunks: one chunk per call on one thread, and one batch on all cores.
- `accuracy`: cosine similarity of every library and precision to the
  model2vec-rs f32 output.

go-potion and encoders are skipped, with a notice, when they cannot be
built. encoders runs batches of fewer than 128 texts on one thread; the
report marks corpora where that applies. Results are written to
`target/bench-suite/results/<timestamp>/` (`report.md`, `results.json`).

## Comparing with Python model2vec

```sh
uv run scripts/python_compare.py
```

Reproduces the Performance table in the main README. The script rebuilds
the pinned Wikipedia corpus under `target/python-compare/` (checked against
a SHA-256 digest), uses each non-empty line as one text, and runs Rust and
Python in alternating processes for `--threads 1,all`. Python uses its
default process pool (`use_multiprocessing=True`) only for `all`; pass
`--no-python-multiprocessing` to turn it off there too. It also reports how
far the Rust embeddings are from the Python ones. Pass `--corpus` and
`--model` to compare on other data or models.

## Int8 models

To benchmark int8 storage, quantize models with Python model2vec and pass
the output directories to `--models`:

```python
from model2vec import StaticModel

for model_id in ["minishlab/potion-base-8M", "minishlab/potion-retrieval-32M"]:
    model = StaticModel.from_pretrained(model_id, quantize_to="int8")
    model.save_pretrained(f"target/bench-compare/int8-models/{model_id.split('/')[1]}")
```

## Runtime int8 quantization

`quantize.rs` compares the f32 model, a separately loaded copy of it (a
control that should come out at 1.00x), the `quantize_int8` model, and
optionally a model2vec int8 model, all in one process:

```sh
cargo bench --bench quantize -- corpus=target/bench-compare/big.txt \
    model=minishlab/potion-base-8M int8=target/bench-compare/int8-models/potion-base-8M
```

Each round runs every variant once, cycling through all orderings so each
variant follows every other equally often. It reports throughput, the
speedup over f32 with a 95% bootstrap CI, and cosine similarity to the f32
output. Set `RAYON_NUM_THREADS=1` for single-thread numbers.


## Retrieval quality (NanoBEIR)

Run this before every release:

```sh
uv run scripts/nanobeir.py
```

The script downloads the 13 NanoBEIR datasets (`zeta-alpha-ai/Nano*`) into
`target/nanobeir/data/`, builds `nanobeir.rs`, and scores every model in
`benches/nanobeir_baseline.json` (NDCG@10, cosine similarity, binary
relevance, scored like `pytrec_eval`). It fails if a model's mean score
drops by more than `--tolerance` (default 0.002) or any single dataset
drops by more than `--dataset-tolerance` (default 0.01). Scoring all
baseline models takes about fifteen seconds once the data and models are
cached.

A model suffixed with `:int8` is scored after `StaticModel::quantize_int8`;
one suffixed with `:model2vec-int8` is first quantized to int8 by Python
model2vec (pinned in the script) and saved under `target/nanobeir/models/`.
To score other models, pass `--models`. When a change is meant to alter
scores, review the deltas and record them with `--update-baseline`
(optionally restricted with `--models`). The baseline scores match Python
model2vec scored with `pytrec_eval` to within 2e-4.
