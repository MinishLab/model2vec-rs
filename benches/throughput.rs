// Throughput benchmark for one (model, workload, mode) configuration, driven
// by `scripts/bench_compare.py`. Uses only the public API so the same file
// builds against any revision.
//
//   cargo bench --bench throughput -- corpus=big.txt model=minishlab/potion-base-8M \
//       workload=queries mode=batch reps=5
//
// Prints one `CONFIG` line, one `CHECKSUM` line (a hash of the bits of every
// output value, to confirm two revisions compute the same embeddings), and
// one `SAMPLE <seconds>` line per timed pass over the workload.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use model2vec_rs::model::StaticModel;

fn checksum(embeddings: &[Vec<f32>]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in embeddings {
        h = (h ^ v.len() as u64).wrapping_mul(0x100000001b3);
        for x in v {
            h = (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn main() {
    let args: HashMap<String, String> = env::args()
        .skip(1)
        .filter_map(|a| a.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())))
        .collect();
    let arg = |k: &str, default: &str| args.get(k).cloned().unwrap_or_else(|| default.to_string());
    let corpus = args
        .get("corpus")
        .expect("usage: throughput corpus=<file> [model=..] [workload=queries|docs] [mode=batch|single] [reps=N]");
    let model_id = arg("model", "minishlab/potion-base-8M");
    let workload = arg("workload", "queries");
    let mode = arg("mode", "batch");
    let reps: usize = arg("reps", "5").parse().expect("reps must be an integer");

    let (words_per_text, texts_per_pass) = match (workload.as_str(), mode.as_str()) {
        ("queries", "batch") => (8, 100_000),
        ("queries", "single") => (8, 20_000),
        ("docs", "batch") => (256, 4_000),
        ("docs", "single") => (256, 1_000),
        other => panic!("unknown workload/mode {other:?}"),
    };
    let text = fs::read_to_string(corpus).unwrap_or_else(|e| panic!("failed to read {corpus}: {e}"));
    let words: Vec<&str> = text.split_whitespace().collect();
    assert!(!words.is_empty(), "corpus {corpus} has no words");
    let texts: Vec<String> = words
        .chunks(words_per_text)
        .cycle()
        .take(texts_per_pass)
        .map(|c| c.join(" "))
        .collect();

    let model = StaticModel::from_pretrained(&model_id, None, None, None)
        .unwrap_or_else(|e| panic!("failed to load {model_id}: {e}"));

    let run = || -> Vec<Vec<f32>> {
        match mode.as_str() {
            "batch" => model.encode(&texts),
            _ => texts.iter().map(|t| model.encode_single(t)).collect(),
        }
    };

    let warm = run();
    let dim = warm.first().map_or(0, Vec::len);
    println!(
        "CONFIG model={model_id} workload={workload} mode={mode} texts={} dim={dim}",
        texts.len()
    );
    println!("CHECKSUM {:016x}", checksum(&warm));
    drop(warm);

    for _ in 0..reps {
        let start = Instant::now();
        black_box(run());
        println!("SAMPLE {:.9}", start.elapsed().as_secs_f64());
    }
}
