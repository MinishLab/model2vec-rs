mod common;
use approx::assert_relative_eq;
use common::load_test_model;
use common::load_test_model_vocab_quantized;
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
