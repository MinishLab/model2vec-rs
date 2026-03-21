#![allow(dead_code)]
use model2vec_rs::model::StaticModel;
use std::fs;
use tempfile::TempDir;

pub fn load_test_model() -> StaticModel {
    assert_loads("tests/fixtures/test-model-float32", None)
}

pub fn load_test_model_vocab_quantized() -> StaticModel {
    assert_loads("tests/fixtures/test-model-vocab-quantized", None)
}

pub fn assert_loads(path: &str, subfolder: Option<&str>) -> StaticModel {
    StaticModel::from_pretrained(path, None, None, subfolder)
        .unwrap_or_else(|e| panic!("failed to load model at {path}: {e}"))
}

pub fn encode_with_model(path: &str) -> Vec<f32> {
    let model = assert_loads(path, None);
    let out = model.encode(&["hello world".to_string()]);
    assert_eq!(out.len(), 1);
    out.into_iter().next().unwrap()
}

pub fn embedding_norm(model: &StaticModel, text: &str) -> f32 {
    let emb = model.encode(&[text.to_string()]);
    emb[0].iter().map(|&x| x * x).sum::<f32>().sqrt()
}

const ST_CONFIG: &str = r#"{"normalize": true}"#;

fn copy_st_blobs(dir: &std::path::Path) {
    for file in ["model.safetensors", "tokenizer.json"] {
        fs::copy(
            format!("tests/fixtures/test-model-sentence-transformers/{file}"),
            dir.join(file),
        )
        .expect("copy fixture blob");
    }
}

pub fn temp_st_dir(modules_json: Option<&str>) -> TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    copy_st_blobs(dir.path());
    fs::write(dir.path().join("config_sentence_transformers.json"), ST_CONFIG).expect("write ST config");
    if let Some(content) = modules_json {
        fs::write(dir.path().join("modules.json"), content).expect("write modules.json");
    }
    dir
}

/// Both configs present: `config.json` has `normalize: false`,
/// `config_sentence_transformers.json` has `normalize: true`.
pub fn temp_both_configs_dir() -> TempDir {
    let dir = temp_st_dir(None);
    fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"model2vec","normalize":false}"#,
    )
    .expect("write config.json");
    dir
}

pub fn temp_nested_st_dir() -> TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = dir.path().join("some/path");
    let emb_dir = base.join("0_StaticEmbedding");
    fs::create_dir_all(&emb_dir).expect("create nested dir");
    copy_st_blobs(&emb_dir);
    fs::write(base.join("config_sentence_transformers.json"), ST_CONFIG).expect("write nested ST config");
    dir
}
