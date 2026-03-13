mod common;
use common::{assert_loads, embedding_norm, load_test_model, temp_both_configs_dir, temp_nested_st_dir, temp_st_dir};
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
        // Native model2vec
        ("tests/fixtures/test-model-float32", None),
        // Sentence-transformers root layout
        ("tests/fixtures/test-model-sentence-transformers", None),
        // 0_StaticEmbedding subfolder (auto-detected)
        ("tests/fixtures/test-model-static-embedding", None),
        // 0_StaticEmbedding via explicit subfolder arg
        ("tests/fixtures/test-model-static-embedding", Some("0_StaticEmbedding")),
        // Direct path into 0_StaticEmbedding
        ("tests/fixtures/test-model-static-embedding/0_StaticEmbedding", None),
        // Both configs present — ST should win
        (both.path().to_str().unwrap(), None),
        // Nested subfolder path
        (nested.path().to_str().unwrap(), Some("some/path/0_StaticEmbedding")),
        // Direct path into nested 0_StaticEmbedding
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

/// config_sentence_transformers.json (normalize=true) must win over config.json (normalize=false).
#[test]
fn test_both_configs_prefers_sentence_transformers() {
    let dir = temp_both_configs_dir();
    let model = assert_loads(dir.path().to_str().unwrap(), None);
    let norm = embedding_norm(&model, "hello world");
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "expected unit norm (ST config wins), got {norm}"
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

/// modules.json Normalize stage controls normalization when config lacks the key
#[test]
fn test_normalize_from_modules_json() {
    const WITH_NORMALIZE: &str = r#"[
        {"idx":0,"name":"0","path":".","type":"sentence_transformers.models.StaticEmbedding"},
        {"idx":1,"name":"1","path":"1_Normalize","type":"sentence_transformers.models.Normalize"}
    ]"#;
    const WITHOUT_NORMALIZE: &str = r#"[
        {"idx":0,"name":"0","path":".","type":"sentence_transformers.models.StaticEmbedding"}
    ]"#;

    let dir_norm = temp_st_dir(Some(WITH_NORMALIZE));
    let dir_no_norm = temp_st_dir(Some(WITHOUT_NORMALIZE));

    // Overwrite with a config that has no "normalize" key so modules.json is the source of truth.
    let no_key_config = r#"{"model_type":"model2vec"}"#;
    std::fs::write(dir_norm.path().join("config_sentence_transformers.json"), no_key_config).unwrap();
    std::fs::write(
        dir_no_norm.path().join("config_sentence_transformers.json"),
        no_key_config,
    )
    .unwrap();

    let model_norm = assert_loads(dir_norm.path().to_str().unwrap(), None);
    let model_no_norm = assert_loads(dir_no_norm.path().to_str().unwrap(), None);

    let norm_normalized = embedding_norm(&model_norm, "hello world");
    let norm_unnormalized = embedding_norm(&model_no_norm, "hello world");

    assert!(
        (norm_normalized - 1.0).abs() < 1e-5,
        "normalized model should have unit norm, got {norm_normalized}"
    );
    assert!(
        norm_unnormalized > norm_normalized,
        "un-normalized model should have larger norm"
    );
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
