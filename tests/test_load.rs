use approx::assert_relative_eq;
use model2vec_rs::model::StaticModel;

fn encode_hello(path: &str) -> Vec<f32> {
    // Helper function to load the model and encode "hello world"
    let model = StaticModel::from_pretrained(path, None)
        .expect(&format!("Failed to load model at {}", path));
    let out = model.encode(&["hello world".to_string()]);
    assert_eq!(out.len(), 1);
    out.into_iter().next().unwrap()
}

#[test]
fn quantized_models_match_float32() {
    // Compare quantized models against the float32 model
    let base = "tests/fixtures/test-model-float32";
    let ref_emb = encode_hello(base);

    for quant in &["float16", "int8"] {
        let path = format!("tests/fixtures/test-model-{}", quant);
        let emb = encode_hello(&path);

        assert_eq!(emb.len(), ref_emb.len());

        for (a, b) in ref_emb.iter().zip(emb.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-1);
        }
    }
}
