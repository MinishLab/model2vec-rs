# Benchmarks

- `encode.rs`: latency percentiles of `StaticModel::encode` for one revision.
- `throughput.rs`: throughput of one (model, workload, mode) configuration.
  Uses only the public API, so it builds against any revision. Normally run
  through `scripts/bench_compare.py`, not directly.

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
