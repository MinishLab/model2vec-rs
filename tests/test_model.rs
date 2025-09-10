mod common;
use approx::assert_relative_eq;
use common::load_test_model;
use common::load_test_model_vocab_quantized;
use model2vec_rs::model::StaticModel;
use std::fs;



#[test]
fn test_encode_matches_python_model2vec() {
    // Load the test model
    let model = load_test_model();

    // Define the short and long text inputs
    let long_text = vec!["hello"; 1000].join(" ");
    let short_text = "hello world".to_string();
    let cases = vec![
        ("tests/fixtures/embeddings_short.json", vec![short_text]),
        ("tests/fixtures/embeddings_long.json", vec![long_text]),
    ];

    for (fixture_path, inputs) in cases {
        // Read and parse the Python‐generated embedding fixture
        let fixture =
            fs::read_to_string(fixture_path).unwrap_or_else(|_| panic!("Fixture not found: {}", fixture_path));
        let expected: Vec<Vec<f32>> = serde_json::from_str(&fixture).expect("Failed to parse fixture");

        // Encode with the Rust model
        let output = model.encode(&inputs);

        // Sanity checks
        assert_eq!(
            output.len(),
            expected.len(),
            "number of sentences mismatch for {}",
            fixture_path
        );
        assert_eq!(
            output[0].len(),
            expected[0].len(),
            "vector dimensionality mismatch for {}",
            fixture_path
        );

        // Element‐wise comparison
        for (o, e) in output[0].iter().zip(&expected[0]) {
            assert_relative_eq!(o, e, max_relative = 1e-5);
        }
    }
}



#[test]
fn test_encode_matches_python_model2vec_vocab_quantized() {
    // Load the test model
    let model = load_test_model_vocab_quantized();

    // Define the short and long text inputs
    let long_text = vec!["hello"; 1000].join(" ");
    let cases = vec![
        ("tests/fixtures/embeddings_vocab_quantized.json", vec![long_text]),
    ];

    for (fixture_path, inputs) in cases {
        // Read and parse the Python‐generated embedding fixture
        let fixture =
            fs::read_to_string(fixture_path).unwrap_or_else(|_| panic!("Fixture not found: {}", fixture_path));
        let expected: Vec<Vec<f32>> = serde_json::from_str(&fixture).expect("Failed to parse fixture");

        // Encode with the Rust model
        let output = model.encode(&inputs);

        // Sanity checks
        assert_eq!(
            output.len(),
            expected.len(),
            "number of sentences mismatch for {}",
            fixture_path
        );
        assert_eq!(
            output[0].len(),
            expected[0].len(),
            "vector dimensionality mismatch for {}",
            fixture_path
        );

        // Element‐wise comparison
        for (o, e) in output[0].iter().zip(&expected[0]) {
            assert_relative_eq!(o, e, max_relative = 1e-5);
        }
    }
}

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
