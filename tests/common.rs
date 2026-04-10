#![allow(dead_code)]
use model2vec_rs::model::StaticModel;
use std::{fs, path::Path};
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

fn copy_model_blobs(source: &Path, target: &Path) {
    for file in ["model.safetensors", "tokenizer.json"] {
        fs::copy(source.join(file), target.join(file)).expect("copy fixture blob");
    }
}

pub fn temp_layout_dir(model_source: &str, model_target: &str, configs: &[(&str, &str)]) -> TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_dir = if model_target.is_empty() {
        dir.path().to_path_buf()
    } else {
        dir.path().join(model_target)
    };
    fs::create_dir_all(&model_dir).expect("create model dir");
    copy_model_blobs(Path::new(model_source), &model_dir);

    for (config_rel, contents) in configs {
        let config_path = dir.path().join(config_rel);
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent).expect("create config dir");
        }
        fs::write(config_path, contents).expect("write config");
    }

    dir
}
