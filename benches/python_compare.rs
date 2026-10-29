// Rust side of the comparison with Python model2vec, driven by
// `scripts/python_compare.py`.
//
//   cargo bench --bench python_compare -- corpus=wiki-en.txt model=minishlab/potion-base-8M \
//       reps=5 out=embeddings.f32
//
// Every non-empty line of the corpus is one text, and each timed pass encodes
// all of them with one `encode` call. Prints one `LOAD <seconds>` line, one
// `TEXTS <n>` line and one `SAMPLE <seconds>` line per timed pass. With
// `out=`, writes the embeddings of the warm-up pass as little-endian f32,
// row-major.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use model2vec_rs::model::StaticModel;

fn main() {
    let args: HashMap<String, String> = env::args()
        .skip(1)
        .filter_map(|a| a.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())))
        .collect();
    let corpus = args
        .get("corpus")
        .expect("usage: python_compare corpus=<file> [model=..] [reps=N] [out=<file>]");
    let model_id = args.get("model").map_or("minishlab/potion-base-8M", String::as_str);
    let reps: usize = args
        .get("reps")
        .map_or(5, |r| r.parse().expect("reps must be an integer"));

    let text = fs::read_to_string(corpus).unwrap_or_else(|e| panic!("failed to read {corpus}: {e}"));
    let texts: Vec<String> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(String::from)
        .collect();
    assert!(!texts.is_empty(), "corpus {corpus} has no non-empty lines");

    let start = Instant::now();
    let model = StaticModel::from_pretrained(model_id, None, None, None)
        .unwrap_or_else(|e| panic!("failed to load {model_id}: {e}"));
    println!("LOAD {:.6}", start.elapsed().as_secs_f64());
    println!("TEXTS {}", texts.len());

    let warm = model.encode(&texts);
    if let Some(out) = args.get("out") {
        let bytes: Vec<u8> = warm.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
        fs::write(out, bytes).unwrap_or_else(|e| panic!("failed to write {out}: {e}"));
    }
    drop(warm);

    for _ in 0..reps {
        let start = Instant::now();
        black_box(model.encode(&texts));
        println!("SAMPLE {:.9}", start.elapsed().as_secs_f64());
    }
}
