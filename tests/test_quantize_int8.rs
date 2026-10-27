//! `StaticModel::quantize_int8`: per-row int8 quantization of f32 models.
//!
//! Exactness is checked against `from_owned_i8` given the same int8 values
//! and per-token weights equal to the token weight times the row scale,
//! which performs the same float operations.

use model2vec_rs::model::StaticModel;
use tokenizers::Tokenizer;

const WORDPIECE_TOKENIZER: &str = "tests/fixtures/test-model-float32/tokenizer.json";
const UNIGRAM_TOKENIZER: &str = "tests/fixtures/test-model-vocab-quantized/tokenizer.json";
const MAPPED_ROWS: usize = 97;

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

fn corpus() -> Vec<String> {
    let mut texts: Vec<String> = [
        "",
        "hello world",
        "The quick brown fox jumps over the lazy dog.",
        "café résumé naïve Ünïcödé",
        "你好世界，这是一个测试。",
        "[MASK] literal [UNK] tokens",
        "unbelievably antidisestablishmentarianism",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    texts.push(vec!["hello"; 600].join(" "));
    texts.push("Static embeddings map every token to a vector. ".repeat(12));
    texts
}

fn reference_quantize(table: &[f32], dim: usize) -> (Vec<i8>, Vec<f32>) {
    let mut data = Vec::new();
    let mut scales = Vec::new();
    for row in table.chunks_exact(dim) {
        let max = row.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let scale = max / 127.0;
        for &v in row {
            data.push(if scale > 0.0 {
                (v / scale).round().clamp(-127.0, 127.0) as i8
            } else {
                0
            });
        }
        scales.push(scale);
    }
    (data, scales)
}

struct Case {
    tokenizer: Tokenizer,
    dim: usize,
    normalize: bool,
    table: Vec<f32>,
    weights: Option<Vec<f32>>,
    mapping: Option<Vec<usize>>,
}

impl Case {
    fn new(tokenizer_path: &str, dim: usize, (weights, mapping, normalize): (bool, bool, bool)) -> Self {
        let tokenizer = Tokenizer::from_file(tokenizer_path).expect("load fixture tokenizer");
        let vocab = tokenizer.get_vocab_size(true);
        let mut rng = Rng(0xD1B54A32D192ED03 ^ (dim as u64) << 4 ^ (weights as u64) << 1 ^ mapping as u64);
        let rows = if mapping { MAPPED_ROWS } else { vocab };
        let mut table: Vec<f32> = (0..rows * dim).map(|_| rng.signed()).collect();
        table[dim..2 * dim].fill(0.0);
        let weights = weights.then(|| (0..vocab).map(|_| rng.next_f32() * 2.0).collect());
        let mapping = mapping.then(|| (0..vocab).map(|_| rng.below(MAPPED_ROWS)).collect());
        Case {
            tokenizer,
            dim,
            normalize,
            table,
            weights,
            mapping,
        }
    }

    fn label(&self) -> String {
        format!(
            "dim={} weights={} mapping={} normalize={}",
            self.dim,
            self.weights.is_some(),
            self.mapping.is_some(),
            self.normalize
        )
    }

    fn f32_model(&self) -> StaticModel {
        StaticModel::from_owned(
            self.tokenizer.clone(),
            self.table.clone(),
            self.table.len() / self.dim,
            self.dim,
            self.normalize,
            self.weights.clone(),
            self.mapping.clone(),
        )
        .unwrap_or_else(|e| panic!("failed to build f32 model for {}: {e}", self.label()))
    }

    fn oracle(&self) -> StaticModel {
        let (data, scales) = reference_quantize(&self.table, self.dim);
        let vocab = self.tokenizer.get_vocab_size(true);
        let token_scales = (0..vocab)
            .map(|tok| {
                let row = self.mapping.as_ref().map_or(tok, |m| m[tok]);
                let weight = self.weights.as_ref().map_or(1.0, |w| w[tok]);
                scales.get(row).copied().unwrap_or(1.0) * weight
            })
            .collect();
        StaticModel::from_owned_i8(
            self.tokenizer.clone(),
            data,
            self.table.len() / self.dim,
            self.dim,
            self.normalize,
            Some(token_scales),
            self.mapping.clone(),
        )
        .unwrap_or_else(|e| panic!("failed to build oracle for {}: {e}", self.label()))
    }
}

fn assert_bit_identical(label: &str, got: &[Vec<f32>], want: &[Vec<f32>]) {
    assert_eq!(got.len(), want.len(), "{label}: output count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.len(), w.len(), "{label}: dimension of text {i}");
        if let Some(j) = g.iter().zip(w).position(|(a, b)| a.to_bits() != b.to_bits()) {
            panic!("{label}: text {i} element {j} differs: got {} want {}", g[j], w[j]);
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na = a.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    let nb = b.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    if na == 0.0 && nb == 0.0 { 1.0 } else { dot / (na * nb) }
}

const COMBOS: [(bool, bool, bool); 8] = [
    (false, false, true),
    (true, false, false),
    (false, true, true),
    (true, true, false),
    (false, false, false),
    (true, false, true),
    (false, true, false),
    (true, true, true),
];

fn check(tokenizer_path: &str, dim: usize, combo: (bool, bool, bool)) {
    let texts = corpus();
    let case = Case::new(tokenizer_path, dim, combo);
    let f32_model = case.f32_model();
    let quantized = case.f32_model().quantize_int8();
    let oracle = case.oracle();
    for max_length in [Some(512), Some(7), None] {
        let label = format!("{} max_length={max_length:?}", case.label());
        assert_bit_identical(
            &label,
            &quantized.encode_with_args(&texts, max_length, 1024),
            &oracle.encode_with_args(&texts, max_length, 1024),
        );
    }
    let singles = |m: &StaticModel| texts.iter().map(|t| m.encode_single(t)).collect::<Vec<_>>();
    assert_bit_identical(&case.label(), &singles(&quantized), &singles(&oracle));

    let cos: Vec<f64> = quantized
        .encode(&texts)
        .iter()
        .zip(&f32_model.encode(&texts))
        .map(|(a, b)| cosine(a, b))
        .collect();
    let mean = cos.iter().sum::<f64>() / cos.len() as f64;
    let min = cos.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        mean >= 0.9999 && min >= 0.999,
        "{}: cosine to f32 mean {mean} min {min}",
        case.label()
    );
}

#[test]
fn wordpiece_quantize_matches_reference_all_options() {
    for dim in [64, 300] {
        for combo in COMBOS {
            check(WORDPIECE_TOKENIZER, dim, combo);
        }
    }
}

#[test]
fn wordpiece_quantize_matches_reference_all_dims() {
    for (i, dim) in [1, 2, 63, 128, 256, 384, 512, 768, 1024].into_iter().enumerate() {
        check(WORDPIECE_TOKENIZER, dim, COMBOS[i % COMBOS.len()]);
    }
}

#[test]
fn unigram_quantize_matches_reference() {
    for (i, dim) in [2, 64, 256, 1000].into_iter().enumerate() {
        check(UNIGRAM_TOKENIZER, dim, COMBOS[(2 * i) % COMBOS.len()]);
        check(UNIGRAM_TOKENIZER, dim, COMBOS[(2 * i + 1) % COMBOS.len()]);
    }
}

#[test]
fn quantize_leaves_int8_models_unchanged() {
    let dir = "tests/fixtures/test-model-int8";
    let model = StaticModel::from_pretrained(dir, None, None, None).expect("load int8 fixture");
    let texts = corpus();
    let want = model.encode(&texts);
    assert_bit_identical("int8 fixture", &model.quantize_int8().encode(&texts), &want);
}

#[test]
fn quantizing_a_shared_model_leaves_the_original_alone() {
    let case = Case::new(WORDPIECE_TOKENIZER, 256, (true, false, true));
    let original = case.f32_model();
    let texts = corpus();
    let before = original.encode(&texts);
    let quantized = original.clone().quantize_int8();
    assert_bit_identical("original after quantizing a clone", &original.encode(&texts), &before);
    assert_bit_identical(
        "quantized shared model",
        &quantized.encode(&texts),
        &case.f32_model().quantize_int8().encode(&texts),
    );
}
