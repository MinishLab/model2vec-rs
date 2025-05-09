use approx::assert_relative_eq;
use std::fs;
use serde_json::Value;
use model2vec_rust::inference::StaticModel;

#[test]
fn test_encode_hello_against_fixture() {
    let fixture = fs::read_to_string("tests/fixtures/embeddings.json")
        .expect("Fixture not found");
    let expected: Vec<Vec<f32>> = serde_json::from_str(&fixture)
        .expect("Failed to parse fixture");

    let model = StaticModel::from_pretrained("minishlab/potion-base-2M").unwrap();
    let output = model.encode(&["hello world".to_string()]);

    assert_eq!(output.len(), expected.len());
    assert_eq!(output[0].len(), expected[0].len());
    for (o, e) in output[0].iter().zip(expected[0].iter()) {
        assert_relative_eq!(o, e, max_relative = 1e-5);
    }
}