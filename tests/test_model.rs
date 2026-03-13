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

/// Test loading a model in sentence-transformers root layout (config_sentence_transformers.json)
#[test]
fn test_load_sentence_transformers_layout() {
    let model = StaticModel::from_pretrained(
        "tests/fixtures/test-model-sentence-transformers",
        None,
        None,
        None,
    )
    .expect("should load sentence-transformers layout");
    let emb = model.encode(&["hello world".to_string()]);
    assert_eq!(emb.len(), 1);
    assert!(!emb[0].is_empty());
}

/// Test loading a model in 0_StaticEmbedding subfolder layout
#[test]
fn test_load_static_embedding_layout() {
    let model = StaticModel::from_pretrained(
        "tests/fixtures/test-model-static-embedding",
        None,
        None,
        None,
    )
    .expect("should load 0_StaticEmbedding layout");
    let emb = model.encode(&["hello world".to_string()]);
    assert_eq!(emb.len(), 1);
    assert!(!emb[0].is_empty());
}

/// Fix 1: subfolder pointing directly at the 0_StaticEmbedding directory
/// Config lives one level up; loader should still succeed.
#[test]
fn test_load_static_embedding_via_subfolder() {
    // repo_or_path = parent, subfolder = "0_StaticEmbedding"
    let model = StaticModel::from_pretrained(
        "tests/fixtures/test-model-static-embedding",
        None,
        None,
        Some("0_StaticEmbedding"),
    )
    .expect("should load when subfolder='0_StaticEmbedding'");
    assert!(!model.encode(&["hello".to_string()])[0].is_empty());

    // repo_or_path points directly at the subfolder
    let model2 = StaticModel::from_pretrained(
        "tests/fixtures/test-model-static-embedding/0_StaticEmbedding",
        None,
        None,
        None,
    )
    .expect("should load when path points directly at 0_StaticEmbedding");
    assert!(!model2.encode(&["hello".to_string()])[0].is_empty());
}

/// Fix 2: both config.json and config_sentence_transformers.json present — ST wins
/// (embedding.weight key must be found even though config.json is also on disk).
#[test]
fn test_load_both_configs_prefers_sentence_transformers() {
    let model = StaticModel::from_pretrained(
        "tests/fixtures/test-model-both-configs",
        None,
        None,
        None,
    )
    .expect("should load when both config files are present");
    assert!(!model.encode(&["hello".to_string()])[0].is_empty());
}

/// Fix 3a: no normalize key in config, but modules.json has a Normalize stage → normalize=true
#[test]
fn test_normalize_from_modules_json_true() {
    let model = StaticModel::from_pretrained(
        "tests/fixtures/test-model-modules-normalize",
        None,
        None,
        None,
    )
    .unwrap();
    let emb = model.encode(&["hello world".to_string()])[0].clone();
    let norm: f32 = emb.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
}

/// Fix 3b: no normalize key in config, modules.json has NO Normalize stage → normalize=false
#[test]
fn test_normalize_from_modules_json_false() {
    let model_no_norm = StaticModel::from_pretrained(
        "tests/fixtures/test-model-modules-no-normalize",
        None,
        None,
        None,
    )
    .unwrap();
    let emb_no = model_no_norm.encode(&["hello world".to_string()])[0].clone();
    let norm_no: f32 = emb_no.iter().map(|&x| x * x).sum::<f32>().sqrt();

    let model_norm = StaticModel::from_pretrained(
        "tests/fixtures/test-model-modules-normalize",
        None,
        None,
        None,
    )
    .unwrap();
    let emb_norm = model_norm.encode(&["hello world".to_string()])[0].clone();
    let norm_norm: f32 = emb_norm.iter().map(|&x| x * x).sum::<f32>().sqrt();

    assert!((norm_norm - 1.0).abs() < 1e-5, "normalized model should have unit norm");
    assert!(norm_no > norm_norm, "un-normalized model should have larger norm");
}

/// Sentence-transformers and model2vec layouts with the same weights should give identical embeddings
#[test]
fn test_sentence_transformers_matches_model2vec() {
    let model_m2v =
        StaticModel::from_pretrained("tests/fixtures/test-model-float32", None, None, None).unwrap();
    let model_st = StaticModel::from_pretrained(
        "tests/fixtures/test-model-sentence-transformers",
        None,
        None,
        None,
    )
    .unwrap();
    let sentences = vec!["hello".to_string(), "world test sentence".to_string()];
    let emb_m2v = model_m2v.encode(&sentences);
    let emb_st = model_st.encode(&sentences);
    for (a, b) in emb_m2v.iter().zip(emb_st.iter()) {
        for (&x, &y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "embeddings should match: {x} vs {y}");
        }
    }
}

/// Test that a path missing all known layouts gives a helpful error
#[test]
fn test_load_invalid_path_error() {
    let result = StaticModel::from_pretrained("tests/fixtures", None, None, None);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("no valid model layout"), "error should mention layout: {msg}");
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
