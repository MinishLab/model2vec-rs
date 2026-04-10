mod common;
use common::load_test_model;
use model2vec_rs::model::StaticModel;
use std::fs;

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

/// Test from_borrowed constructor (zero-copy path)
#[test]
fn test_from_borrowed() {
    use safetensors::SafeTensors;
    use tokenizers::Tokenizer;

    let path = "tests/fixtures/test-model-float32";
    let tokenizer = Tokenizer::from_file(format!("{path}/tokenizer.json")).unwrap();
    let bytes = fs::read(format!("{path}/model.safetensors")).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let tensor = tensors.tensor("embeddings").unwrap();
    let [rows, cols]: [usize; 2] = tensor.shape().try_into().unwrap();
    let floats: Vec<f32> = tensor
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    // Leak to get 'static lifetime (fine for tests)
    let floats: &'static [f32] = Box::leak(floats.into_boxed_slice());

    let model = StaticModel::from_borrowed(tokenizer, floats, rows, cols, true, None, None).unwrap();
    let emb = model.encode_single("hello");
    assert!(!emb.is_empty());
}

#[test]
fn test_from_bytes_matches_from_pretrained_for_local_model() {
    let path = "tests/fixtures/test-model-float32";
    let from_path = StaticModel::from_pretrained(path, None, None, None).unwrap();
    let from_bytes = StaticModel::from_bytes(
        fs::read(format!("{path}/tokenizer.json")).unwrap(),
        fs::read(format!("{path}/model.safetensors")).unwrap(),
        fs::read(format!("{path}/config.json")).unwrap(),
        None,
    )
    .unwrap();

    let query = "hello world";
    let path_embedding = from_path.encode_single(query);
    let bytes_embedding = from_bytes.encode_single(query);

    assert_eq!(path_embedding.len(), bytes_embedding.len());
    for (left, right) in path_embedding.iter().zip(bytes_embedding.iter()) {
        assert!(
            (left - right).abs() < 1e-6,
            "expected byte-loaded model to match path-loaded model"
        );
    }
}

#[cfg(not(feature = "hf-hub"))]
#[test]
fn test_from_pretrained_remote_requires_hf_hub_feature() {
    let err = StaticModel::from_pretrained("minishlab/potion-base-2M", None, None, None).unwrap_err();
    assert!(
        err.to_string().contains("hf-hub"),
        "expected remote loading without hf-hub to mention the missing feature"
    );
}

#[cfg(all(feature = "hf-hub", feature = "local-only"))]
#[test]
fn test_from_pretrained_remote_disallowed_by_local_only_feature() {
    let err = StaticModel::from_pretrained("minishlab/potion-base-2M", None, None, None).unwrap_err();
    assert!(
        err.to_string().contains("local-only"),
        "expected remote loading with local-only to mention the local-only restriction"
    );
}
