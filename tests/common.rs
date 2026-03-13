#![allow(dead_code)]
use model2vec_rs::model::StaticModel;
use std::fs;
use tempfile::TempDir;

/// Load the small float32 test model from fixtures
pub fn load_test_model() -> StaticModel {
    assert_loads("tests/fixtures/test-model-float32", None)
}

/// Load the vocab quantized test model from fixtures
pub fn load_test_model_vocab_quantized() -> StaticModel {
    assert_loads("tests/fixtures/test-model-vocab-quantized", None)
}

/// Load a model from `path` with an optional `subfolder`, panicking on failure.
pub fn assert_loads(path: &str, subfolder: Option<&str>) -> StaticModel {
    StaticModel::from_pretrained(path, None, None, subfolder)
        .unwrap_or_else(|e| panic!("failed to load model at {path}: {e}"))
}

/// Encode `text` with a model loaded from `path`/`subfolder`, return the embedding.
pub fn encode_with_model(path: &str) -> Vec<f32> {
    let model = assert_loads(path, None);
    let out = model.encode(&["hello world".to_string()]);
    assert_eq!(out.len(), 1);
    out.into_iter().next().unwrap()
}

/// L2 norm of `model.encode(&[text])`.
pub fn embedding_norm(model: &StaticModel, text: &str) -> f32 {
    let emb = model.encode(&[text.to_string()]);
    emb[0].iter().map(|&x| x * x).sum::<f32>().sqrt()
}

// ---------------------------------------------------------------------------
// Temp-dir fixture builders
// ---------------------------------------------------------------------------

const ST_CONFIG: &str = r#"{"normalize": true}"#;
const PLAIN_CONFIG: &str = r#"{"model_type":"model2vec","normalize":true,"hidden_dim":64}"#;

/// Resolve a source file from the sentence-transformers fixture.
fn st_src(file: &str) -> String {
    format!("tests/fixtures/test-model-sentence-transformers/{file}")
}

/// Copy the two binary blobs (model + tokenizer) from the ST fixture into `dir`.
fn copy_st_blobs(dir: &std::path::Path) {
    for file in ["model.safetensors", "tokenizer.json"] {
        fs::copy(st_src(file), dir.join(file)).expect("copy fixture blob");
    }
}

/// Build a temp dir that looks like a sentence-transformers root layout,
/// using `modules_json` as the content of `modules.json` (pass `None` to omit).
pub fn temp_st_dir(modules_json: Option<&str>) -> TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    copy_st_blobs(dir.path());
    fs::write(dir.path().join("config_sentence_transformers.json"), ST_CONFIG)
        .expect("write ST config");
    if let Some(content) = modules_json {
        fs::write(dir.path().join("modules.json"), content).expect("write modules.json");
    }
    dir
}

/// Build a temp dir that has BOTH `config.json` and `config_sentence_transformers.json`.
pub fn temp_both_configs_dir() -> TempDir {
    let dir = temp_st_dir(None);
    fs::write(dir.path().join("config.json"), PLAIN_CONFIG).expect("write config.json");
    dir
}

/// Build a temp dir with a nested `some/path/0_StaticEmbedding/` layout:
/// - `some/path/config_sentence_transformers.json`
/// - `some/path/0_StaticEmbedding/{model.safetensors,tokenizer.json}`
pub fn temp_nested_st_dir() -> TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().join("some/path");
    let emb_dir = base.join("0_StaticEmbedding");
    fs::create_dir_all(&emb_dir).expect("create nested dir");
    copy_st_blobs(&emb_dir);
    fs::write(base.join("config_sentence_transformers.json"), ST_CONFIG)
        .expect("write nested ST config");
    dir
}
