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

