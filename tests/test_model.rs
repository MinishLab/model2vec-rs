mod common;
use common::load_test_model;
use approx::assert_relative_eq;
use std::fs;
use model2vec_rs::model::StaticModel;

/// Test that encoding "hello world" matches the Python-generated fixture
#[test]
fn test_encode_matches_python_model2vec() {
    let fixture = fs::read_to_string("tests/fixtures/embeddings.json")
        .expect("Fixture not found");
    let expected: Vec<Vec<f32>> = serde_json::from_str(&fixture)
        .expect("Failed to parse fixture");
    let model = load_test_model();
    let output = model.encode(&["hello world".to_string()]);
    assert_eq!(output.len(), expected.len());
    assert_eq!(output[0].len(), expected[0].len());
    for (o, e) in output[0].iter().zip(expected[0].iter()) {
        assert_relative_eq!(o, e, max_relative = 1e-5);
    }
}

/// Test that encoding an empty input slice yields an empty Vec
#[test]
fn test_encode_empty_input() {
    let model = load_test_model();
    let embs: Vec<Vec<f32>> = model.encode(&[]);
    assert!(embs.is_empty(), "Expected no embeddings for empty input");
}

/// Test encoding a single empty sentence produces a zero vector with no NaNs
#[test]
fn test_encode_empty_sentence() {
    let model = load_test_model();
    let embs = model.encode(&["".to_string()]);
    assert_eq!(embs.len(), 1);
    let vec = &embs[0];
    assert!(vec.iter().all(|&x| x == 0.0), "All entries should be zero");
}

/// Test override of `normalize` flag in from_pretrained
#[test]
fn test_normalization_flag_override() {
    // first load with normalize = true (default in config)
    let model_norm = StaticModel::from_pretrained(
        "tests/fixtures/test-model-float32", None, None, None
    ).unwrap();
    let emb_norm = model_norm.encode(&["test sentence".to_string()])[0].clone();
    let norm_norm = emb_norm.iter().map(|&x| x*x).sum::<f32>().sqrt();

    // now load with normalize = false override
    let model_no_norm = StaticModel::from_pretrained(
        "tests/fixtures/test-model-float32", None, Some(false), None
    ).unwrap();
    let emb_no = model_no_norm.encode(&["test sentence".to_string()])[0].clone();
    let norm_no = emb_no.iter().map(|&x| x*x).sum::<f32>().sqrt();

    // normalized version should have unit length, override should give larger norm
    assert!((norm_norm - 1.0).abs() < 1e-5, "Normalized vector should have unit norm");
    assert!(norm_no > norm_norm, "Without normalization override, norm should be larger");
}
