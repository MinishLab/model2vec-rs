mod common;
use approx::assert_relative_eq;
use common::encode_with_model;

#[test]
fn quantized_models_match_float32() {
    // Compare quantized models against the float32 model
    let base = "tests/fixtures/test-model-float32";
    let ref_emb = encode_with_model(base);

    for quant in &["float16", "int8"] {
        let path = format!("tests/fixtures/test-model-{}", quant);
        let emb = encode_with_model(&path);

        assert_eq!(emb.len(), ref_emb.len());

        for (a, b) in ref_emb.iter().zip(emb.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-1);
        }
    }
}
