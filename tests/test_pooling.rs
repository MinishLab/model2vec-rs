//! Bit-exact parity between `StaticModel::encode*` and a straightforward
//! reference implementation of tokenize -> truncate -> drop unk -> weighted
//! mean -> normalize, across embedding dimensions (including every
//! dimension with a specialized pooling loop), weights/mapping combinations,
//! normalization, truncation, and both tokenizer paths.

use std::sync::OnceLock;

use model2vec_rs::model::StaticModel;
use serde_json::Value;
use tokenizers::{Tokenizer, models::ModelWrapper};

const WORDPIECE_TOKENIZER: &str = "tests/fixtures/test-model-float32/tokenizer.json";
const UNIGRAM_TOKENIZER: &str = "tests/fixtures/test-model-vocab-quantized/tokenizer.json";
const MAX_DIM: usize = 1024;
const MAPPED_ROWS: usize = 97;
const DIMS: &[usize] = &[1, 2, 63, 64, 128, 256, 300, 384, 512, 768, 1000, 1024];

struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn signed(&mut self) -> f32 {
        let magnitude = 10f32.powi((self.next_f32() * 4.0) as i32 - 2);
        (self.next_f32() * 2.0 - 1.0) * magnitude
    }

    fn below(&mut self, n: usize) -> usize {
        ((self.next_f32() * n as f32) as usize).min(n - 1)
    }
}

fn shared_table() -> &'static [f32] {
    static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        (0..30_000 * MAX_DIM).map(|_| rng.signed()).collect()
    })
}

fn corpus() -> Vec<String> {
    let mut texts: Vec<String> = [
        "",
        "   ",
        "hello world",
        "Hello, World! How are you doing today?",
        "The quick brown fox jumps over the lazy dog.",
        "café résumé naïve Ünïcödé façade",
        "你好世界，这是一个测试。",
        "emoji 🎉🚀 and symbols ∑∫√ ☃",
        "[MASK] literal [UNK] tokens [CLS] and [SEP]",
        "tabs\tand\nnewlines\r\nmixed",
        "\u{0}\u{1}control\u{7f}chars\u{fffd}",
        "unbelievably antidisestablishmentarianism pseudopseudohypoparathyroidism",
        "x",
        "!!!???...",
        "1234567890 3.14159 1e-9",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    texts.push("a".repeat(150));
    texts.push(format!("start {} end", "z".repeat(101)));
    texts.push(vec!["hello"; 600].join(" "));
    texts.push(
        "Static embeddings map every token to a vector and average them. "
            .repeat(12)
            .trim_end()
            .to_string(),
    );
    texts.push("ab".repeat(400));
    texts
}

#[derive(Clone, Copy, Debug)]
enum Extras {
    None,
    Weights,
    Mapping,
    Both,
}

struct Case {
    tokenizer: Tokenizer,
    dim: usize,
    extras: Extras,
    normalize: bool,
    embeddings: &'static [f32],
    weights: Option<Vec<f32>>,
    mapping: Option<Vec<usize>>,
}

impl Case {
    fn new(tokenizer: &Tokenizer, dim: usize, extras: Extras, normalize: bool) -> Self {
        let vocab = tokenizer.get_vocab_size(true);
        let mut rng = Rng(0xD1B54A32D192ED03 ^ (dim as u64) << 8 ^ extras as u64);
        let weights = matches!(extras, Extras::Weights | Extras::Both)
            .then(|| (0..vocab).map(|_| rng.next_f32() * 2.0).collect::<Vec<f32>>());
        let mapping = matches!(extras, Extras::Mapping | Extras::Both)
            .then(|| (0..vocab).map(|_| rng.below(MAPPED_ROWS)).collect::<Vec<usize>>());
        let rows = if mapping.is_some() { MAPPED_ROWS } else { vocab };
        let table = shared_table();
        let offset = rng.below(table.len() - rows * dim);
        Case {
            tokenizer: tokenizer.clone(),
            dim,
            extras,
            normalize,
            embeddings: &table[offset..offset + rows * dim],
            weights,
            mapping,
        }
    }

    fn model(&self) -> StaticModel {
        let leak = |v: &Option<Vec<f32>>| v.as_ref().map(|v| &*Vec::leak(v.clone()));
        StaticModel::from_borrowed(
            self.tokenizer.clone(),
            self.embeddings,
            self.embeddings.len() / self.dim,
            self.dim,
            self.normalize,
            leak(&self.weights),
            self.mapping.as_ref().map(|m| &*Vec::leak(m.clone())),
        )
        .unwrap_or_else(|e| panic!("failed to build model for {}: {e}", self.label()))
    }

    fn label(&self) -> String {
        format!("dim={} extras={:?} normalize={}", self.dim, self.extras, self.normalize)
    }

    fn reference(&self, token_ids: &[Vec<usize>]) -> Vec<Vec<f32>> {
        token_ids
            .iter()
            .map(|ids| {
                let mut sum = vec![0.0_f32; self.dim];
                for &tok in ids {
                    let row = self.mapping.as_ref().and_then(|m| m.get(tok)).copied().unwrap_or(tok);
                    let scale = self.weights.as_ref().and_then(|w| w.get(tok)).copied().unwrap_or(1.0);
                    for (j, s) in sum.iter_mut().enumerate() {
                        *s += self.embeddings[row * self.dim + j] * scale;
                    }
                }
                let denom = ids.len().max(1) as f32;
                for x in &mut sum {
                    *x /= denom;
                }
                if self.normalize {
                    let norm = sum.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-12);
                    for x in &mut sum {
                        *x /= norm;
                    }
                }
                sum
            })
            .collect()
    }
}

/// Token IDs per text after character truncation, tokenization, token
/// truncation and unk removal.
fn reference_ids(tokenizer: &Tokenizer, texts: &[String], max_length: Option<usize>) -> Vec<Vec<usize>> {
    let unk = unk_id(tokenizer);
    let median = median_token_length(tokenizer);
    let truncated: Vec<String> = texts
        .iter()
        .map(|t| match max_length {
            Some(m) => t.chars().take(m * median).collect(),
            None => t.clone(),
        })
        .collect();
    tokenizer
        .encode_batch_fast::<String>(truncated, false)
        .expect("reference tokenization failed")
        .iter()
        .map(|enc| {
            let mut ids: Vec<usize> = enc.get_ids().iter().map(|&i| i as usize).collect();
            if let Some(m) = max_length {
                ids.truncate(m);
            }
            ids.retain(|&id| Some(id) != unk);
            ids
        })
        .collect()
}

fn unk_id(tokenizer: &Tokenizer) -> Option<usize> {
    let id = match tokenizer.get_model() {
        ModelWrapper::WordPiece(m) => tokenizer.token_to_id(&m.unk_token),
        ModelWrapper::Unigram(m) => serde_json::to_value(m)
            .unwrap()
            .get("unk_id")
            .and_then(Value::as_u64)
            .map(|id| id as u32),
        other => panic!("unexpected tokenizer model in test: {other:?}"),
    };
    id.map(|id| id as usize)
}

fn median_token_length(tokenizer: &Tokenizer) -> usize {
    let mut lens: Vec<usize> = tokenizer.get_vocab(false).keys().map(|t| t.len()).collect();
    lens.sort_unstable();
    lens.get(lens.len() / 2).copied().unwrap_or(1)
}

fn assert_bit_identical(case: &Case, what: &str, got: &[Vec<f32>], want: &[Vec<f32>], texts: &[String]) {
    assert_eq!(got.len(), want.len(), "{} {what}: output count", case.label());
    for ((g, w), text) in got.iter().zip(want).zip(texts) {
        let text: String = text.chars().take(60).collect();
        assert_eq!(g.len(), w.len(), "{} {what}: dimension for {text:?}", case.label());
        let mismatch = g.iter().zip(w).position(|(a, b)| a.to_bits() != b.to_bits());
        if let Some(j) = mismatch {
            panic!(
                "{} {what}: element {j} differs for {text:?}: got {} want {}",
                case.label(),
                g[j],
                w[j]
            );
        }
    }
}

const MAX_LENGTHS: [Option<usize>; 4] = [Some(512), Some(7), Some(0), None];

fn check_all(tokenizer_path: &str, dims: &[usize]) {
    let tokenizer = Tokenizer::from_file(tokenizer_path).expect("load fixture tokenizer");
    let texts = corpus();
    let ids: Vec<Vec<Vec<usize>>> = MAX_LENGTHS
        .iter()
        .map(|&m| reference_ids(&tokenizer, &texts, m))
        .collect();
    for &dim in dims {
        for extras in [Extras::None, Extras::Weights, Extras::Mapping, Extras::Both] {
            for normalize in [true, false] {
                let case = Case::new(&tokenizer, dim, extras, normalize);
                let model = case.model();

                for (max_length, ids) in MAX_LENGTHS.iter().zip(&ids) {
                    let want = case.reference(ids);
                    for batch_size in [1, 1024] {
                        let got = model.encode_with_args(&texts, *max_length, batch_size);
                        let what = format!("encode_with_args(max_length={max_length:?}, batch_size={batch_size})");
                        assert_bit_identical(&case, &what, &got, &want, &texts);
                    }
                    if *max_length == Some(512) {
                        assert_bit_identical(&case, "encode", &model.encode(&texts), &want, &texts);
                        let singles: Vec<Vec<f32>> = texts.iter().map(|t| model.encode_single(t)).collect();
                        assert_bit_identical(&case, "encode_single", &singles, &want, &texts);
                    }
                }
            }
        }
    }
}

#[test]
fn wordpiece_pooling_matches_reference() {
    check_all(WORDPIECE_TOKENIZER, DIMS);
}

#[test]
fn unigram_pooling_matches_reference() {
    check_all(UNIGRAM_TOKENIZER, &[2, 64, 256, 1000]);
}

#[test]
fn interleaved_lengths_reuse_buffers_correctly() {
    let tokenizer = Tokenizer::from_file(WORDPIECE_TOKENIZER).expect("load fixture tokenizer");
    let case = Case::new(&tokenizer, 256, Extras::None, true);
    let model = case.model();
    let long = vec!["hello"; 600].join(" ");
    let texts: Vec<String> = (0..64)
        .map(|i| {
            if i % 3 == 0 {
                long.clone()
            } else {
                format!("short text {i}")
            }
        })
        .collect();
    let want = case.reference(&reference_ids(&tokenizer, &texts, Some(512)));
    let singles: Vec<Vec<f32>> = texts.iter().map(|t| model.encode_single(t)).collect();
    assert_bit_identical(&case, "encode_single", &singles, &want, &texts);
    assert_bit_identical(&case, "encode", &model.encode(&texts), &want, &texts);
}

#[test]
fn clones_share_results_across_threads() {
    let tokenizer = Tokenizer::from_file(WORDPIECE_TOKENIZER).expect("load fixture tokenizer");
    let case = Case::new(&tokenizer, 512, Extras::Weights, true);
    let model = case.model();
    let texts = corpus();
    let want = case.reference(&reference_ids(&tokenizer, &texts, Some(512)));
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let model = model.clone();
            let texts = texts.clone();
            std::thread::spawn(move || {
                (0..5)
                    .map(|_| texts.iter().map(|t| model.encode_single(t)).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    for handle in handles {
        for got in handle.join().expect("encoding thread panicked") {
            assert_bit_identical(&case, "threaded encode_single", &got, &want, &texts);
        }
    }
}
