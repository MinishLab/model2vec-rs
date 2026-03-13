use anyhow::{anyhow, Context, Result};
use half::f16;
use hf_hub::api::sync::{Api, ApiRepo};
use ndarray::Array2;
use safetensors::{tensor::Dtype, SafeTensors};
use serde_json::Value;
use std::{env, fs, path::{Path, PathBuf}};
use tokenizers::Tokenizer;

/// Resolved file paths and embedding tensor key for a detected model layout.
struct ResolvedPaths {
    tokenizer_path: PathBuf,
    model_path: PathBuf,
    config_path: PathBuf,
    /// Explicitly downloaded/located `modules.json`; `None` for local paths (derived at read time).
    modules_path: Option<PathBuf>,
    /// Safetensors key for the embedding matrix.
    embedding_key: &'static str,
}

/// Try to detect a valid model layout in a local folder.
///
/// Tries layouts in order:
/// 1. model2vec: `config.json` + `model.safetensors` + `tokenizer.json`
/// 2. Sentence Transformers root: `config_sentence_transformers.json` + `model.safetensors` + `tokenizer.json`
/// 3. 0_StaticEmbedding subfolder: `config_sentence_transformers.json` at root, model files under `0_StaticEmbedding/`
/// 4. Config-in-parent fallback: model files at `folder`, `config_sentence_transformers.json` one level up
///    (handles `subfolder = "0_StaticEmbedding"` or direct paths into the embedding subfolder).
fn resolve_local(folder: &Path) -> Option<ResolvedPaths> {
    let t = folder.join("tokenizer.json");
    let m = folder.join("model.safetensors");

    // 1. Native model2vec: config.json + model.safetensors + tokenizer.json
    //    (config_sentence_transformers.json must NOT coexist to avoid misidentifying ST exports)
    let c = folder.join("config.json");
    let c_st = folder.join("config_sentence_transformers.json");
    if t.exists() && m.exists() && c.exists() && !c_st.exists() {
        return Some(ResolvedPaths { tokenizer_path: t, model_path: m, config_path: c, modules_path: None, embedding_key: "embeddings" });
    }

    // 2. Sentence Transformers root: config_sentence_transformers.json + model.safetensors + tokenizer.json
    //    This also covers model2vec models re-exported via sentence-transformers (both configs present).
    if c_st.exists() && t.exists() && m.exists() {
        return Some(ResolvedPaths {
            tokenizer_path: t,
            model_path: m,
            config_path: c_st,
            modules_path: None,
            embedding_key: "embedding.weight",
        });
    }

    // 3. 0_StaticEmbedding subfolder: config_sentence_transformers.json at root, model files in subfolder
    let sub = folder.join("0_StaticEmbedding");
    let t_sub = sub.join("tokenizer.json");
    let m_sub = sub.join("model.safetensors");
    if c_st.exists() && t_sub.exists() && m_sub.exists() {
        return Some(ResolvedPaths {
            tokenizer_path: t_sub,
            model_path: m_sub,
            config_path: c_st,
            modules_path: None,
            embedding_key: "embedding.weight",
        });
    }

    // 4. Config-in-parent fallback: the caller pointed directly at the embedding subfolder
    //    (e.g. subfolder="0_StaticEmbedding" or repo_or_path = ".../0_StaticEmbedding").
    //    The config lives one level up in the actual ST model root.
    if t.exists() && m.exists() {
        if let Some(parent) = folder.parent() {
            let c_parent = parent.join("config_sentence_transformers.json");
            if c_parent.exists() {
                return Some(ResolvedPaths {
                    tokenizer_path: t,
                    model_path: m,
                    config_path: c_parent,
                    modules_path: None,
                    embedding_key: "embedding.weight",
                });
            }
        }
    }

    None
}

/// Strip the last path component from a Hub prefix (which always ends with `/`).
///
/// Returns the parent prefix, also ending with `/`, or an empty string for the repo root.
///
/// * `"0_StaticEmbedding/"` → `""`
/// * `"some/path/0_StaticEmbedding/"` → `"some/path/"`
fn parent_of_prefix(prefix: &str) -> String {
    let trimmed = prefix.trim_end_matches('/');
    match Path::new(trimmed).parent() {
        Some(p) if !p.as_os_str().is_empty() => format!("{}/", p.display()),
        _ => String::new(),
    }
}

/// Download the tokenizer and model files for a sentence-transformers Hub repo.
///
/// Tries the root layout first (`{prefix}tokenizer.json` + `{prefix}model.safetensors`),
/// then falls back to the `0_StaticEmbedding` subfolder.
fn hub_st_files(repo: &ApiRepo, prefix: &str) -> Result<(PathBuf, PathBuf)> {
    if let (Ok(t), Ok(m)) = (
        repo.get(&format!("{prefix}tokenizer.json")),
        repo.get(&format!("{prefix}model.safetensors")),
    ) {
        return Ok((t, m));
    }
    let t = repo
        .get(&format!("{prefix}0_StaticEmbedding/tokenizer.json"))
        .context("tokenizer.json not found (tried root and 0_StaticEmbedding/)")?;
    let m = repo
        .get(&format!("{prefix}0_StaticEmbedding/model.safetensors"))
        .context("model.safetensors not found (tried root and 0_StaticEmbedding/)")?;
    Ok((t, m))
}

/// Read the normalize flag from a model's config and optional `modules.json`.
///
/// Resolution order:
/// 1. `normalize` key in the config file.
/// 2. Presence of a `sentence_transformers.models.Normalize` entry in `modules.json`.
///    For local paths `explicit_modules` is `None` and the file is looked up next to
///    the config; for Hub downloads the already-fetched path is passed explicitly.
/// 3. Default: `true`.
fn read_normalize(config_path: &Path, explicit_modules: Option<&Path>) -> bool {
    // 1. Check config file
    if let Ok(f) = std::fs::File::open(config_path) {
        if let Ok(cfg) = serde_json::from_reader::<_, Value>(f) {
            if let Some(v) = cfg.get("normalize").and_then(Value::as_bool) {
                return v;
            }
        }
    }

    // 2. Check modules.json for a Normalize pipeline stage.
    //    Use explicit path when provided (Hub), otherwise derive from config's directory (local).
    let derived;
    let modules_path: Option<&Path> = match explicit_modules {
        Some(p) => Some(p),
        None => {
            derived = config_path
                .parent()
                .unwrap_or_else(|| Path::new(""))
                .join("modules.json");
            if derived.exists() { Some(&derived) } else { None }
        }
    };
    if let Some(mp) = modules_path {
        if let Ok(f) = std::fs::File::open(mp) {
            if let Ok(Value::Array(modules)) = serde_json::from_reader::<_, Value>(f) {
                return modules.iter().any(|module| {
                    module
                        .get("type")
                        .and_then(Value::as_str)
                        .map(|t| t.contains("Normalize"))
                        .unwrap_or(false)
                });
            }
        }
    }

    // 3. Default
    true
}

/// Static embedding model for Model2Vec
#[derive(Debug, Clone)]
pub struct StaticModel {
    tokenizer: Tokenizer,
    embeddings: Array2<f32>,
    weights: Option<Vec<f32>>,
    token_mapping: Option<Vec<usize>>,
    normalize: bool,
    median_token_length: usize,
    unk_token_id: Option<usize>,
}

impl StaticModel {
    /// Load a Model2Vec model from a local folder or the HuggingFace Hub.
    ///
    /// Supports three layouts (tried in order):
    /// - **model2vec**: `config.json` + `model.safetensors` + `tokenizer.json`
    /// - **sentence-transformers**: `config_sentence_transformers.json` + `model.safetensors` + `tokenizer.json`
    /// - **0_StaticEmbedding**: `config_sentence_transformers.json` at root, model files under `0_StaticEmbedding/`
    ///
    /// # Arguments
    /// * `repo_or_path` - HuggingFace repo ID or local path to the model folder.
    /// * `token` - Optional HuggingFace token for authenticated downloads.
    /// * `normalize` - Optional flag to normalize embeddings (default from config file).
    /// * `subfolder` - Optional subfolder within the repo or path to look for model files.
    pub fn from_pretrained<P: AsRef<Path>>(
        repo_or_path: P,
        token: Option<&str>,
        normalize: Option<bool>,
        subfolder: Option<&str>,
    ) -> Result<Self> {
        // If provided, set HF token for authenticated downloads
        if let Some(tok) = token {
            env::set_var("HF_HUB_TOKEN", tok);
        }

        // Locate model files by trying layouts in priority order
        let ResolvedPaths { tokenizer_path: tok_path, model_path: mdl_path, config_path: cfg_path, modules_path: mod_path, embedding_key } = {
            let base = repo_or_path.as_ref();
            if base.exists() {
                let folder = subfolder.map(|s| base.join(s)).unwrap_or_else(|| base.to_path_buf());
                resolve_local(&folder).ok_or_else(|| anyhow!(
                    "no valid model layout found in {folder:?}. \
                     Tried: model2vec (config.json), sentence-transformers \
                     (config_sentence_transformers.json), and 0_StaticEmbedding subfolder."
                ))?
            } else {
                let api = Api::new().context("hf-hub API init failed")?;
                let repo = api.model(repo_or_path.as_ref().to_string_lossy().into_owned());
                let prefix = subfolder.map(|s| format!("{s}/")).unwrap_or_default();

                // modules.json is optional; download it now so read_normalize can inspect it.
                let modules = repo.get(&format!("{prefix}modules.json")).ok();

                // 1. config.json exists — check whether config_sentence_transformers.json also
                //    exists. If both are present, ST wins (mirrors the local resolver's guard).
                if let Ok(c_m2v) = repo.get(&format!("{prefix}config.json")) {
                    if let Ok(c_st) = repo.get(&format!("{prefix}config_sentence_transformers.json")) {
                        // Both configs → sentence-transformers layout takes precedence
                        let (t, m) = hub_st_files(&repo, &prefix)?;
                        ResolvedPaths { tokenizer_path: t, model_path: m, config_path: c_st, modules_path: modules, embedding_key: "embedding.weight" }
                    } else {
                        // Only config.json → native model2vec
                        let t = repo.get(&format!("{prefix}tokenizer.json")).context("tokenizer.json missing")?;
                        let m = repo.get(&format!("{prefix}model.safetensors")).context("model.safetensors missing")?;
                        ResolvedPaths { tokenizer_path: t, model_path: m, config_path: c_m2v, modules_path: modules, embedding_key: "embeddings" }
                    }
                }
                // 2. Only config_sentence_transformers.json (root or 0_StaticEmbedding subfolder)
                else if let Ok(c_st) = repo.get(&format!("{prefix}config_sentence_transformers.json")) {
                    let (t, m) = hub_st_files(&repo, &prefix)?;
                    ResolvedPaths { tokenizer_path: t, model_path: m, config_path: c_st, modules_path: modules, embedding_key: "embedding.weight" }
                }
                // 3. Config-in-parent fallback: the subfolder points directly at the model
                //    files and the config lives one level up (mirrors resolve_local layout 4).
                //    Use parent_of_prefix so nested paths like "some/path/0_StaticEmbedding/"
                //    look in "some/path/", not the repo root.
                else if !prefix.is_empty() {
                    let parent_prefix = parent_of_prefix(&prefix);
                    let parent_modules = repo.get(&format!("{parent_prefix}modules.json")).ok();
                    if let Ok(c) = repo.get(&format!("{parent_prefix}config_sentence_transformers.json")) {
                        let t = repo.get(&format!("{prefix}tokenizer.json"))
                            .context("tokenizer.json not found in subfolder")?;
                        let m = repo.get(&format!("{prefix}model.safetensors"))
                            .context("model.safetensors not found in subfolder")?;
                        ResolvedPaths { tokenizer_path: t, model_path: m, config_path: c, modules_path: parent_modules, embedding_key: "embedding.weight" }
                    } else {
                        return Err(anyhow!(
                            "no valid model layout found on HuggingFace Hub for '{}'. \
                             Tried: model2vec (config.json), sentence-transformers \
                             (config_sentence_transformers.json), 0_StaticEmbedding subfolder, \
                             and config-in-parent fallback.",
                            repo_or_path.as_ref().display()
                        ));
                    }
                } else {
                    return Err(anyhow!(
                        "no valid model layout found on HuggingFace Hub for '{}'. \
                         Tried: model2vec (config.json), sentence-transformers \
                         (config_sentence_transformers.json), and 0_StaticEmbedding subfolder.",
                        repo_or_path.as_ref().display()
                    ));
                }
            }
        };

        // Load the tokenizer
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;

        // Median-token-length hack for pre-truncation
        let mut lens: Vec<usize> = tokenizer.get_vocab(false).keys().map(|tk| tk.len()).collect();
        lens.sort_unstable();
        let median_token_length = lens.get(lens.len() / 2).copied().unwrap_or(1);

        // Read normalize default: config file first, then modules.json, then true.
        // Pass the explicitly downloaded modules path for Hub models; local models derive it.
        let normalize = normalize.unwrap_or_else(|| read_normalize(&cfg_path, mod_path.as_deref()));

        // Serialize the tokenizer to JSON, then parse it and get the unk_token
        let spec_json = tokenizer
            .to_string(false)
            .map_err(|e| anyhow!("tokenizer -> JSON failed: {e}"))?;
        let spec: Value = serde_json::from_str(&spec_json)?;
        let unk_token = spec
            .get("model")
            .and_then(|m| m.get("unk_token"))
            .and_then(Value::as_str)
            .unwrap_or("[UNK]");
        // If the unk token isn't in the vocabulary, treat it as absent rather than erroring.
        let unk_token_id = tokenizer.token_to_id(unk_token).map(|id| id as usize);

        // Load the safetensors
        let model_bytes = fs::read(&mdl_path).context("failed to read model.safetensors")?;
        let safet = SafeTensors::deserialize(&model_bytes).context("failed to parse safetensors")?;
        // Try the layout-specific key first, then all known fallback keys.
        let tensor = safet
            .tensor(embedding_key)
            .or_else(|_| safet.tensor("embeddings"))
            .or_else(|_| safet.tensor("embedding.weight"))
            .or_else(|_| safet.tensor("0"))
            .with_context(|| {
                format!("embedding tensor not found (tried '{embedding_key}', 'embeddings', 'embedding.weight', '0')")
            })?;

        let [rows, cols]: [usize; 2] = tensor.shape().try_into().context("embedding tensor is not 2‑D")?;
        let raw = tensor.data();
        let dtype = tensor.dtype();

        // Decode into f32
        let floats: Vec<f32> = match dtype {
            Dtype::F32 => raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            Dtype::F16 => raw
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32())
                .collect(),
            Dtype::I8 => raw.iter().map(|&b| f32::from(b as i8)).collect(),
            other => return Err(anyhow!("unsupported tensor dtype: {other:?}")),
        };
        let embeddings = Array2::from_shape_vec((rows, cols), floats).context("failed to build embeddings array")?;

        // Load optional weights for vocabulary quantization
        let weights = match safet.tensor("weights") {
            Ok(t) => {
                let raw = t.data();
                let v: Vec<f32> = match t.dtype() {
                    Dtype::F64 => raw
                        .chunks_exact(8)
                        .map(|b| f64::from_le_bytes(b.try_into().unwrap()) as f32)
                        .collect(),
                    Dtype::F32 => raw
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect(),
                    Dtype::F16 => raw
                        .chunks_exact(2)
                        .map(|b| half::f16::from_le_bytes(b.try_into().unwrap()).to_f32())
                        .collect(),
                    other => return Err(anyhow!("unsupported weights dtype: {:?}", other)),
                };
                Some(v)
            }
            Err(_) => None,
        };

        // Load optional token mapping for vocabulary quantization
        let token_mapping = match safet.tensor("mapping") {
            Ok(t) => {
                let raw = t.data();
                let v: Vec<usize> = raw
                    .chunks_exact(4)
                    .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as usize)
                    .collect();
                Some(v)
            }
            Err(_) => None,
        };

        Ok(Self {
            tokenizer,
            embeddings,
            weights,
            token_mapping,
            normalize,
            median_token_length,
            unk_token_id,
        })
    }

    /// Char-level truncation to max_tokens * median_token_length
    fn truncate_str(s: &str, max_tokens: usize, median_len: usize) -> &str {
        let max_chars = max_tokens.saturating_mul(median_len);
        match s.char_indices().nth(max_chars) {
            Some((byte_idx, _)) => &s[..byte_idx],
            None => s,
        }
    }

    /// Encode texts into embeddings.
    ///
    /// # Arguments
    /// * `sentences` - the list of sentences to encode.
    /// * `max_length` - max tokens per text.
    /// * `batch_size` - number of texts per batch.
    pub fn encode_with_args(
        &self,
        sentences: &[String],
        max_length: Option<usize>,
        batch_size: usize,
    ) -> Vec<Vec<f32>> {
        let mut embeddings = Vec::with_capacity(sentences.len());

        // Process in batches
        for batch in sentences.chunks(batch_size) {
            // Truncate each sentence to max_length * median_token_length chars
            let truncated: Vec<&str> = batch
                .iter()
                .map(|text| {
                    max_length
                        .map(|max_tok| Self::truncate_str(text, max_tok, self.median_token_length))
                        .unwrap_or(text.as_str())
                })
                .collect();

            // Tokenize the batch
            let encodings = self
                .tokenizer
                .encode_batch_fast::<String>(
                    // Into<EncodeInput>
                    truncated.into_iter().map(Into::into).collect(),
                    /* add_special_tokens = */ false,
                )
                .expect("tokenization failed");

            // Pool each token-ID list into a single mean vector
            for encoding in encodings {
                let mut token_ids = encoding.get_ids().to_vec();
                // Remove unk tokens if specified
                if let Some(unk_id) = self.unk_token_id {
                    token_ids.retain(|&id| id as usize != unk_id);
                }
                // Truncate to max_length if specified
                if let Some(max_tok) = max_length {
                    token_ids.truncate(max_tok);
                }
                embeddings.push(self.pool_ids(token_ids));
            }
        }

        embeddings
    }

    /// Default encode: `max_length=512`, `batch_size=1024`
    pub fn encode(&self, sentences: &[String]) -> Vec<Vec<f32>> {
        self.encode_with_args(sentences, Some(512), 1024)
    }

    // / Encode a single sentence into a vector
    pub fn encode_single(&self, sentence: &str) -> Vec<f32> {
        self.encode(&[sentence.to_string()])
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    /// Mean-pool a single token-ID list into a vector
    fn pool_ids(&self, ids: Vec<u32>) -> Vec<f32> {
        let dim = self.embeddings.ncols();
        let mut sum = vec![0.0; dim];
        let mut cnt = 0usize;

        for &id in &ids {
            let tok = id as usize;

            // Remap: row = token_mapping[id] or id
            let row_idx = if let Some(m) = &self.token_mapping {
                *m.get(tok).unwrap_or(&tok)
            } else {
                tok
            };

            // Scale by per-token weight if present
            let scale = if let Some(w) = &self.weights {
                *w.get(tok).unwrap_or(&1.0)
            } else {
                1.0
            };

            let row = self.embeddings.row(row_idx);
            for (i, &v) in row.iter().enumerate() {
                sum[i] += v * scale;
            }
            cnt += 1;
        }

        // Mean pool the embeddings
        let denom = (cnt.max(1)) as f32;
        for x in &mut sum {
            *x /= denom;
        }

        // Normalize the embeddings if required
        if self.normalize {
            let norm = sum.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-12);
            for x in &mut sum {
                *x /= norm;
            }
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::parent_of_prefix;

    #[test]
    fn test_parent_of_prefix_root_subfolder() {
        assert_eq!(parent_of_prefix("0_StaticEmbedding/"), "");
    }

    #[test]
    fn test_parent_of_prefix_nested() {
        assert_eq!(parent_of_prefix("some/path/0_StaticEmbedding/"), "some/path/");
    }

    #[test]
    fn test_parent_of_prefix_single_level() {
        assert_eq!(parent_of_prefix("models/"), "");
    }

    #[test]
    fn test_parent_of_prefix_two_levels() {
        assert_eq!(parent_of_prefix("a/b/"), "a/");
    }
}
