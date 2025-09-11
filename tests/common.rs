#![allow(dead_code)]
use model2vec_rs::model::StaticModel;

/// Load the small float32 test model from fixtures
pub fn load_test_model() -> StaticModel {
    StaticModel::from_pretrained(
        "tests/fixtures/test-model-float32",
        None, // token
        None, // normalize
        None, // subfolder
    )
    .expect("Failed to load test model")
}

/// Load the vocab quantized test model from fixtures
pub fn load_test_model_vocab_quantized() -> StaticModel {
    StaticModel::from_pretrained(
        "tests/fixtures/test-model-vocab-quantized",
        None, // token
        None, // normalize
        None, // subfolder
    )
    .expect("Failed to load test model")
}

pub fn encode_with_model(path: &str) -> Vec<f32> {
    // Helper function to load the model and encode "hello world"
    let model = StaticModel::from_pretrained(path, None, None, None)
        .unwrap_or_else(|e| panic!("Failed to load model at {path}: {e}"));

    let out = model.encode(&["hello world".to_string()]);
    assert_eq!(out.len(), 1);
    out.into_iter().next().unwrap()
}