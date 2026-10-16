// Latency and throughput benchmark for `StaticModel::encode`.
//
//   curl -o big.txt https://norvig.com/big.txt
//   cargo bench --bench encode -- big.txt
//   cargo bench --bench encode -- big.txt minishlab/potion-retrieval-32M 5
//
// The corpus is split into 256-word chunks. Reports per-call latency
// percentiles for one chunk per call and for a sweep of batch sizes.

use std::env;
use std::fs;
use std::hint::black_box;
use std::time::{Duration, Instant};

use model2vec_rs::model::StaticModel;

fn main() {
    // `cargo bench` injects libtest flags (e.g. `--bench`); keep only positionals.
    let pos: Vec<String> = env::args().skip(1).filter(|a| !a.starts_with('-')).collect();
    let corpus = pos
        .first()
        .expect("usage: cargo bench --bench encode -- <corpus.txt> [model] [reps]");
    let model_id = pos.get(1).map_or("minishlab/potion-base-8M", String::as_str);
    let reps: usize = pos.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    let text = fs::read_to_string(corpus).unwrap_or_else(|e| panic!("failed to read {corpus}: {e}"));
    let words: Vec<&str> = text.split_whitespace().collect();
    let chunks: Vec<String> = words.chunks(256).map(|c| c.join(" ")).collect();
    assert!(!chunks.is_empty(), "corpus {corpus} is empty");
    let total_bytes: usize = chunks.iter().map(String::len).sum();

    let model = StaticModel::from_pretrained(model_id, None, None, None)
        .unwrap_or_else(|e| panic!("failed to load {model_id}: {e}"));

    println!(
        "model={model_id}  chunks={}  bytes={total_bytes}  reps={reps}\n",
        chunks.len()
    );
    println!(
        "{:<12} {:>7} {:>11} {:>11} {:>11} {:>10}",
        "batch", "calls", "p50", "p99", "p99.9", "MB/s"
    );

    for bs in [1usize, 32, 256, 1024, chunks.len()] {
        let bs = bs.clamp(1, chunks.len());
        let name = if bs == chunks.len() {
            "all".to_string()
        } else {
            bs.to_string()
        };
        // Warm up once, then time every call.
        for group in chunks.chunks(bs) {
            black_box(model.encode(group));
        }
        let mut samples = Vec::new();
        let mut elapsed = Duration::ZERO;
        for _ in 0..reps {
            for group in chunks.chunks(bs) {
                let start = Instant::now();
                black_box(model.encode(group));
                let took = start.elapsed();
                samples.push(took);
                elapsed += took;
            }
        }
        samples.sort_unstable();
        let mbps = (total_bytes * reps) as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0;
        println!(
            "{name:<12} {:>7} {:>11.1?} {:>11.1?} {:>11.1?} {mbps:>10.2}",
            samples.len(),
            percentile(&samples, 0.50),
            percentile(&samples, 0.99),
            percentile(&samples, 0.999),
        );
    }
}

/// Nearest-rank percentile of sorted `samples`.
fn percentile(samples: &[Duration], p: f64) -> Duration {
    let rank = (samples.len() as f64 * p).ceil() as usize;
    samples[rank.clamp(1, samples.len()) - 1]
}
