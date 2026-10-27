//! Int8 models (from safetensors, `from_owned_i8` or `from_borrowed_i8`) are
//! pooled from their stored i8 values; the output must be bit-identical to
//! loading the same values widened to f32.

use std::collections::HashMap;
use std::fs;

use model2vec_rs::model::StaticModel;
use safetensors::tensor::{Dtype, TensorView, serialize};
use tokenizers::Tokenizer;

const WORDPIECE_TOKENIZER: &str = "tests/fixtures/test-model-float32/tokenizer.json";
const UNIGRAM_TOKENIZER: &str = "tests/fixtures/test-model-vocab-quantized/tokenizer.json";
const MAPPED_ROWS: usize = 97;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
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

struct Case {
    tokenizer_json: Vec<u8>,
    dim: usize,
    normalize: bool,
    table: Vec<i8>,
    weights: Option<Vec<f32>>,
    mapping: Option<Vec<i64>>,
}

impl Case {
    fn new(tokenizer_path: &str, dim: usize, weights: bool, mapping: bool, normalize: bool) -> Self {
        let tokenizer_json = fs::read(tokenizer_path).expect("read fixture tokenizer");
        let vocab = Tokenizer::from_bytes(&tokenizer_json)
            .expect("load fixture tokenizer")
            .get_vocab_size(true);
        let mut rng = Rng(0x9E3779B97F4A7C15 ^ (dim as u64) << 4 ^ (weights as u64) << 1 ^ mapping as u64);
        let rows = if mapping { MAPPED_ROWS } else { vocab };
        let table = (0..rows * dim).map(|_| (rng.next() >> 56) as u8 as i8).collect();
        let weights = weights.then(|| {
            (0..vocab)
                .map(|_| (rng.next() >> 40) as f32 / (1u64 << 23) as f32)
                .collect()
        });
        let mapping = mapping.then(|| (0..vocab).map(|_| rng.below(MAPPED_ROWS) as i64).collect());
        Case {
            tokenizer_json,
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

    fn int8_model(&self) -> StaticModel {
        let rows = self.table.len() / self.dim;
        let table: Vec<u8> = self.table.iter().map(|&v| v as u8).collect();
        let weights: Option<Vec<u8>> = self
            .weights
            .as_ref()
            .map(|w| w.iter().flat_map(|v| v.to_le_bytes()).collect());
        let mapping: Option<Vec<u8>> = self
            .mapping
            .as_ref()
            .map(|m| m.iter().flat_map(|v| v.to_le_bytes()).collect());
        let mut tensors = vec![(
            "embeddings".to_string(),
            TensorView::new(Dtype::I8, vec![rows, self.dim], &table).unwrap(),
        )];
        if let (Some(bytes), Some(w)) = (&weights, &self.weights) {
            tensors.push((
                "weights".to_string(),
                TensorView::new(Dtype::F32, vec![w.len()], bytes).unwrap(),
            ));
        }
        if let (Some(bytes), Some(m)) = (&mapping, &self.mapping) {
            tensors.push((
                "mapping".to_string(),
                TensorView::new(Dtype::I64, vec![m.len()], bytes).unwrap(),
            ));
        }
        let model_bytes = serialize(tensors, &None::<HashMap<String, String>>).expect("serialize safetensors");
        let config = format!("{{\"normalize\": {}}}", self.normalize);
        StaticModel::from_bytes(&self.tokenizer_json, model_bytes, config, None)
            .unwrap_or_else(|e| panic!("failed to load int8 model for {}: {e}", self.label()))
    }

    fn tokenizer(&self) -> Tokenizer {
        Tokenizer::from_bytes(&self.tokenizer_json).expect("load fixture tokenizer")
    }

    fn mapping_usize(&self) -> Option<Vec<usize>> {
        self.mapping.as_ref().map(|m| m.iter().map(|&v| v as usize).collect())
    }

    fn owned_i8_model(&self) -> StaticModel {
        StaticModel::from_owned_i8(
            self.tokenizer(),
            self.table.clone(),
            self.table.len() / self.dim,
            self.dim,
            self.normalize,
            self.weights.clone(),
            self.mapping_usize(),
        )
        .unwrap_or_else(|e| panic!("failed to build owned i8 model for {}: {e}", self.label()))
    }

    fn borrowed_i8_model(&self) -> StaticModel {
        StaticModel::from_borrowed_i8(
            self.tokenizer(),
            Vec::leak(self.table.clone()),
            self.table.len() / self.dim,
            self.dim,
            self.normalize,
            self.weights.clone().map(|w| &*Vec::leak(w)),
            self.mapping_usize().map(|m| &*Vec::leak(m)),
        )
        .unwrap_or_else(|e| panic!("failed to build borrowed i8 model for {}: {e}", self.label()))
    }

    fn f32_model(&self) -> StaticModel {
        StaticModel::from_owned(
            self.tokenizer(),
            self.table.iter().map(|&v| f32::from(v)).collect(),
            self.table.len() / self.dim,
            self.dim,
            self.normalize,
            self.weights.clone(),
            self.mapping_usize(),
        )
        .unwrap_or_else(|e| panic!("failed to build f32 model for {}: {e}", self.label()))
    }
}

fn assert_bit_identical(label: &str, got: &[Vec<f32>], want: &[Vec<f32>]) {
    assert_eq!(got.len(), want.len(), "{label}: output count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.len(), w.len(), "{label}: dimension of text {i}");
        if let Some(j) = g.iter().zip(w).position(|(a, b)| a.to_bits() != b.to_bits()) {
            panic!("{label}: text {i} element {j} differs: int8 {} f32 {}", g[j], w[j]);
        }
    }
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

fn check(tokenizer_path: &str, dim: usize, (weights, mapping, normalize): (bool, bool, bool)) {
    let texts = corpus();
    let case = Case::new(tokenizer_path, dim, weights, mapping, normalize);
    let (int8, f32_model) = (case.int8_model(), case.f32_model());
    for max_length in [Some(512), Some(7), None] {
        let label = format!("{} max_length={max_length:?}", case.label());
        assert_bit_identical(
            &label,
            &int8.encode_with_args(&texts, max_length, 1024),
            &f32_model.encode_with_args(&texts, max_length, 1024),
        );
    }
    let singles = |m: &StaticModel| texts.iter().map(|t| m.encode_single(t)).collect::<Vec<_>>();
    assert_bit_identical(&case.label(), &singles(&int8), &singles(&f32_model));
    let want = f32_model.encode(&texts);
    for (what, model) in [
        ("from_owned_i8", case.owned_i8_model()),
        ("from_borrowed_i8", case.borrowed_i8_model()),
    ] {
        assert_bit_identical(&format!("{} {what}", case.label()), &model.encode(&texts), &want);
    }
}

#[test]
fn wordpiece_int8_matches_widened_f32_all_options() {
    for dim in [64, 300] {
        for combo in COMBOS {
            check(WORDPIECE_TOKENIZER, dim, combo);
        }
    }
}

#[test]
fn wordpiece_int8_matches_widened_f32_all_dims() {
    for (i, dim) in [1, 2, 63, 128, 256, 384, 512, 768, 1024].into_iter().enumerate() {
        check(WORDPIECE_TOKENIZER, dim, COMBOS[i % COMBOS.len()]);
    }
}

#[test]
fn unigram_int8_matches_widened_f32() {
    for (i, dim) in [2, 64, 256, 1000].into_iter().enumerate() {
        check(UNIGRAM_TOKENIZER, dim, COMBOS[(2 * i) % COMBOS.len()]);
        check(UNIGRAM_TOKENIZER, dim, COMBOS[(2 * i + 1) % COMBOS.len()]);
    }
}

#[test]
fn int8_fixture_matches_widened_f32() {
    let dir = "tests/fixtures/test-model-int8";
    let int8 = StaticModel::from_pretrained(dir, None, None, None).expect("load int8 fixture");
    let raw = fs::read(format!("{dir}/model.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&raw).unwrap();
    let tensor = st.tensor("embeddings").unwrap();
    assert_eq!(tensor.dtype(), Dtype::I8);
    let [rows, cols]: [usize; 2] = tensor.shape().try_into().unwrap();
    let widened = StaticModel::from_owned(
        Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap(),
        tensor.data().iter().map(|&b| f32::from(b as i8)).collect(),
        rows,
        cols,
        true,
        None,
        None,
    )
    .unwrap();
    let texts = corpus();
    assert_bit_identical("int8 fixture", &int8.encode(&texts), &widened.encode(&texts));
}
