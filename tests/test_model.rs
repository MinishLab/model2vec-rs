mod common;
use common::{assert_loads, embedding_norm, load_test_model, temp_both_configs_dir, temp_nested_st_dir};
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
    assert!(embs[0].iter().all(|&x| x == 0.0), "All entries should be zero");
}

/// Test that encoding a single sentence returns the correct shape
#[test]
fn test_encode_single() {
    let model = load_test_model();
    let sentence = "hello world";
    let one_d = model.encode_single(sentence);
    let two_d = model.encode(&[sentence.to_string()]);
    assert!(!one_d.is_empty(), "encode_single must return a non-empty 1-D vector");
    assert_eq!(two_d.len(), 1);
    assert_eq!(two_d[0].len(), one_d.len());
}

/// All supported model layouts should load and produce non-empty embeddings
#[test]
fn test_all_layouts_load() {
    let both = temp_both_configs_dir();
    let nested = temp_nested_st_dir();

    let cases: &[(&str, Option<&str>)] = &[
        ("tests/fixtures/test-model-float32", None),
        ("tests/fixtures/test-model-sentence-transformers", None),
        ("tests/fixtures/test-model-static-embedding", None),
        ("tests/fixtures/test-model-static-embedding", Some("0_StaticEmbedding")),
        ("tests/fixtures/test-model-static-embedding/0_StaticEmbedding", None),
        (both.path().to_str().unwrap(), None),
        (nested.path().to_str().unwrap(), Some("some/path/0_StaticEmbedding")),
        (
            &format!("{}/some/path/0_StaticEmbedding", nested.path().display()),
            None,
        ),
    ];

    for &(path, subfolder) in cases {
        let model = assert_loads(path, subfolder);
        let emb = model.encode(&["hello".to_string()]);
        assert!(
            !emb[0].is_empty(),
            "empty embedding for path={path:?} subfolder={subfolder:?}"
        );
    }
}

/// When both config.json and config_sentence_transformers.json are present, native wins
/// (config.json), matching Python model2vec layout resolution order.
/// config.json has normalize=false so embeddings should not be unit-normalized.
#[test]
fn test_both_configs_prefers_native() {
    let dir = temp_both_configs_dir();
    let model = assert_loads(dir.path().to_str().unwrap(), None);
    let norm = embedding_norm(&model, "hello world");
    assert!(
        (norm - 1.0).abs() > 1e-3,
        "expected non-unit norm (native config.json wins with normalize=false), got {norm}"
    );
}

/// ST and native model2vec layouts with the same weights should give identical embeddings
#[test]
fn test_sentence_transformers_matches_model2vec() {
    let model_m2v = StaticModel::from_pretrained("tests/fixtures/test-model-float32", None, None, None).unwrap();
    let model_st =
        StaticModel::from_pretrained("tests/fixtures/test-model-sentence-transformers", None, None, None).unwrap();
    let sentences = vec!["hello".to_string(), "world test sentence".to_string()];
    for (a, b) in model_m2v
        .encode(&sentences)
        .iter()
        .zip(model_st.encode(&sentences).iter())
    {
        for (&x, &y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "embeddings should match: {x} vs {y}");
        }
    }
}

/// Override of the `normalize` flag in from_pretrained works correctly
#[test]
fn test_normalization_flag_override() {
    let model_norm = StaticModel::from_pretrained("tests/fixtures/test-model-float32", None, None, None).unwrap();
    let model_no_norm =
        StaticModel::from_pretrained("tests/fixtures/test-model-float32", None, Some(false), None).unwrap();

    let norm_norm = embedding_norm(&model_norm, "test sentence");
    let norm_no = embedding_norm(&model_no_norm, "test sentence");

    assert!(
        (norm_norm - 1.0).abs() < 1e-5,
        "normalized vector should have unit norm"
    );
    assert!(
        norm_no > norm_norm,
        "without normalization override, norm should be larger"
    );
}

/// A path that matches no known layout returns a helpful error
#[test]
fn test_load_invalid_path_error() {
    let result = StaticModel::from_pretrained("tests/fixtures", None, None, None);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("no valid model layout"),
        "error should mention layout: {msg}"
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

#[cfg(all(not(feature = "hf-hub"), not(feature = "local-only")))]
#[test]
fn test_from_pretrained_remote_requires_hf_hub_feature() {
    let err = StaticModel::from_pretrained("minishlab/potion-base-2M", None, None, None).unwrap_err();
    assert!(
        err.to_string().contains("hf-hub"),
        "expected remote loading without hf-hub to mention the missing feature"
    );
}

#[cfg(feature = "local-only")]
#[test]
fn test_from_pretrained_remote_disallowed_by_local_only_feature() {
    let err = StaticModel::from_pretrained("minishlab/potion-base-2M", None, None, None).unwrap_err();
    assert!(
        err.to_string().contains("local-only"),
        "expected remote loading with local-only to mention the local-only restriction"
    );
}
