// Throughput and accuracy of `StaticModel::quantize_int8` against the f32
// model it was made from, in one process.
//
//   cargo bench --bench quantize -- corpus=big.txt model=minishlab/potion-base-8M \
//       [int8=path/to/model2vec-int8-model] [rounds=24]
//
// Variants: the f32 model, a second, separately loaded copy of it (a control
// whose speedup should be 1.00x), the per-row quantized model, and optionally
// a model quantized to int8 by Python model2vec. Each round times every
// variant once, cycling through every ordering of the variants so each one
// follows every other equally often. Speedups are medians of per-round
// paired ratios against f32, with a 95% bootstrap CI. Accuracy is cosine
// similarity to the f32 output.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use model2vec_rs::model::StaticModel;

fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn bootstrap_ci(ratios: &[f64]) -> (f64, f64) {
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut meds: Vec<f64> = (0..10_000)
        .map(|_| {
            let sample: Vec<f64> = (0..ratios.len())
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ratios[(state % ratios.len() as u64) as usize]
                })
                .collect();
            median(&sample)
        })
        .collect();
    meds.sort_by(f64::total_cmp);
    (meds[250], meds[9_749])
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for rest in permutations(n - 1) {
        for i in 0..=rest.len() {
            let mut p = rest.clone();
            p.insert(i, n - 1);
            out.push(p);
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na = a.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    let nb = b.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    if na == 0.0 && nb == 0.0 { 1.0 } else { dot / (na * nb) }
}

fn main() {
    let args: HashMap<String, String> = env::args()
        .skip(1)
        .filter_map(|a| a.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())))
        .collect();
    let arg = |k: &str, default: &str| args.get(k).cloned().unwrap_or_else(|| default.to_string());
    let corpus = args
        .get("corpus")
        .expect("usage: quantize corpus=<file> [model=..] [int8=<dir>] [rounds=N]");
    let model_id = arg("model", "minishlab/potion-base-8M");

    let orders = permutations(3 + usize::from(args.contains_key("int8")));
    let rounds: usize = arg("rounds", &orders.len().to_string())
        .parse()
        .expect("rounds must be an integer");
    assert!(rounds > 0, "rounds must be at least 1");

    let text = fs::read_to_string(corpus).unwrap_or_else(|e| panic!("failed to read {corpus}: {e}"));
    let words: Vec<&str> = text.split_whitespace().collect();
    assert!(!words.is_empty(), "corpus {corpus} contains no words");
    let make = |n: usize, k: usize| -> Vec<String> { words.chunks(n).cycle().take(k).map(|c| c.join(" ")).collect() };

    let load = |id: &str| {
        StaticModel::from_pretrained(id, None, None, None).unwrap_or_else(|e| panic!("failed to load {id}: {e}"))
    };
    let mut variants = vec![
        ("f32".to_string(), load(&model_id)),
        ("f32 control".to_string(), load(&model_id)),
        ("quantize_int8".to_string(), load(&model_id).quantize_int8()),
    ];
    if let Some(dir) = args.get("int8") {
        variants.push(("model2vec int8".to_string(), load(dir)));
    }

    let accuracy_texts: Vec<String> = make(8, 20_000).into_iter().chain(make(256, 2_000)).collect();
    let reference = variants[0].1.encode(&accuracy_texts);
    println!("model={model_id} dim={} rounds={rounds}", reference[0].len());
    for (name, model) in &variants[2..] {
        let mut cos: Vec<f64> = model
            .encode(&accuracy_texts)
            .iter()
            .zip(&reference)
            .map(|(a, b)| cosine(a, b))
            .collect();
        cos.sort_by(f64::total_cmp);
        let mean = cos.iter().sum::<f64>() / cos.len() as f64;
        println!(
            "ACCURACY {name:<15} cosine to f32: mean {mean:.6}  p1 {:.6}  min {:.6}",
            cos[cos.len() / 100],
            cos[0]
        );
    }

    let workloads = [
        ("queries", "batch", make(8, 100_000)),
        ("queries", "single", make(8, 20_000)),
        ("docs", "batch", make(256, 4_000)),
        ("docs", "single", make(256, 1_000)),
    ];
    for (workload, mode, texts) in &workloads {
        let run = |model: &StaticModel| -> f64 {
            let start = Instant::now();
            match *mode {
                "batch" => {
                    black_box(model.encode(texts));
                }
                _ => {
                    for t in texts {
                        black_box(model.encode_single(t));
                    }
                }
            }
            start.elapsed().as_secs_f64()
        };
        for (_, model) in &variants {
            run(model);
        }
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
        for r in 0..rounds {
            for &i in &orders[r % orders.len()] {
                times[i].push(run(&variants[i].1));
            }
        }
        println!(
            "{workload:<8} {mode:<6} {:<15} {:>10.0} texts/s",
            "f32",
            texts.len() as f64 / median(&times[0])
        );
        for (i, (name, _)) in variants.iter().enumerate().skip(1) {
            let ratios: Vec<f64> = times[0].iter().zip(&times[i]).map(|(a, b)| a / b).collect();
            let (lo, hi) = bootstrap_ci(&ratios);
            let flag = if lo > 1.0 {
                "faster"
            } else if hi < 1.0 {
                "SLOWER"
            } else {
                ""
            };
            println!(
                "{workload:<8} {mode:<6} {name:<15} {:>10.0} texts/s  {:.3}x [{lo:.3}, {hi:.3}] {flag}",
                texts.len() as f64 / median(&times[i]),
                median(&ratios)
            );
        }
    }
}
