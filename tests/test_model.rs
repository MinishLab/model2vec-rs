mod common;
use common::load_test_model;
use model2vec_rs::model::StaticModel;

/// Test that encoding an empty input slice yields an empty output
#[test]
fn test_encode_empty_input() {
    let model = load_test_model();
    let embs: Vec<Vec<f32>> = model.encode(&[]);
    assert!(embs.is_empty(), "Expected no embeddings for empty input");
}

/// Test that encoding a single empty sentence produces a zero vector
#[test]
fn test_encode_empty_sentence() {
    let model = load_test_model();
    let embs = model.encode(&["".to_string()]);
    assert_eq!(embs.len(), 1);
    let vec = &embs[0];
    assert!(vec.iter().all(|&x| x == 0.0), "All entries should be zero");
}

/// Test that encoding a single sentence returns the correct shape
#[test]
fn test_encode_single() {
    let model = load_test_model();
    let sentence = "hello world";

    // Single-sentence helper → 1-D
    let one_d = model.encode_single(sentence);

    // Batch call with a 1-element slice → 2-D wrapper
    let two_d = model.encode(&[sentence.to_string()]);

    // Shape assertions
    assert!(!one_d.is_empty(), "encode_single must return a non-empty 1-D vector");
    assert_eq!(
        two_d.len(),
        1,
        "encode(&[..]) should wrap the result in a Vec with length 1"
    );
    assert_eq!(
        two_d[0].len(),
        one_d.len(),
        "inner vector dimensionality should match encode_single output"
    );
}

/// Test override of `normalize` flag in from_pretrained
#[test]
fn test_normalization_flag_override() {
    // Load with normalize = true (default in config)
    let model_norm = StaticModel::from_pretrained("tests/fixtures/test-model-float32", None, None, None).unwrap();
    let emb_norm = model_norm.encode(&["test sentence".to_string()])[0].clone();
    let norm_norm = emb_norm.iter().map(|&x| x * x).sum::<f32>().sqrt();

    // Load with normalize = false override
    let model_no_norm =
        StaticModel::from_pretrained("tests/fixtures/test-model-float32", None, Some(false), None).unwrap();
    let emb_no = model_no_norm.encode(&["test sentence".to_string()])[0].clone();
    let norm_no = emb_no.iter().map(|&x| x * x).sum::<f32>().sqrt();

    // Normalized version should have unit length, override should give larger norm
    assert!(
        (norm_norm - 1.0).abs() < 1e-5,
        "Normalized vector should have unit norm"
    );
    assert!(
        norm_no > norm_norm,
        "Without normalization override, norm should be larger"
    );
}
