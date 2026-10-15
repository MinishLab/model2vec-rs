use model2vec_rs::model::StaticModel;
use std::{env, fs, hint::black_box, time::Instant};

fn main() {
    let args: Vec<_> = env::args().collect();
    assert!(
        args.len() >= 3,
        "usage: cargo bench --bench load -- TOKENIZER MODEL [REPEATS]"
    );
    let tokenizer = fs::read(&args[1]).expect("read tokenizer");
    let weights = fs::read(&args[2]).expect("read weights");
    let repeats: usize = args.get(3).map(|s| s.parse().expect("integer repeats")).unwrap_or(50);
    assert!(repeats > 0);
    let texts = vec!["Find the function that retries a failed request with exponential backoff.".to_string()];
    let mut rows = Vec::new();
    let mut expected = None;
    for iteration in 0..repeats + 5 {
        let start = Instant::now();
        let model = StaticModel::from_bytes(&tokenizer, &weights, b"{}", Some(true)).expect("load model");
        let load_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let embedding = black_box(model.encode_with_args(&texts, None, 1024));
        let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
        if let Some(ref expected) = expected {
            assert_eq!(&embedding, expected);
        } else {
            expected = Some(embedding);
        }
        if iteration >= 5 {
            rows.push(serde_json::json!({"load_ms": load_ms, "encode_ms": encode_ms}));
        }
    }
    println!("{}", serde_json::json!({"samples": rows, "embedding": expected}));
}
