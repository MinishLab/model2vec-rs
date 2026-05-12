mod common;
use approx::assert_relative_eq;
use common::load_test_model;
use common::load_test_model_vocab_quantized;
use model2vec_rs::model::StaticModel;
use std::fs;

fn check_fixture(model: &StaticModel, fixture_path: &str, inputs: Vec<String>) {
    let fixture = fs::read_to_string(fixture_path).unwrap_or_else(|_| panic!("fixture not found: {fixture_path}"));
    let expected: Vec<Vec<f32>> = serde_json::from_str(&fixture).expect("failed to parse fixture");
    let output = model.encode(&inputs);

    assert_eq!(output.len(), expected.len(), "sentence count mismatch for {fixture_path}");
    assert_eq!(output[0].len(), expected[0].len(), "dimension mismatch for {fixture_path}");
    for (o, e) in output[0].iter().zip(&expected[0]) {
        assert_relative_eq!(o, e, max_relative = 1e-5);
    }
}

#[test]
fn test_encode_matches_python_model2vec() {
    let model = load_test_model();
    let long_text = vec!["hello"; 1000].join(" ");
    check_fixture(&model, "tests/fixtures/embeddings_short.json", vec!["hello world".to_string()]);
    check_fixture(&model, "tests/fixtures/embeddings_long.json", vec![long_text]);
}

#[test]
fn test_encode_matches_python_model2vec_vocab_quantized() {
    let model = load_test_model_vocab_quantized();
    let long_text = vec!["hello"; 1000].join(" ");
    check_fixture(&model, "tests/fixtures/embeddings_vocab_quantized.json", vec![long_text]);
}
