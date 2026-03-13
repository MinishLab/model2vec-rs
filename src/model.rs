use anyhow::{anyhow, Context, Result};
use half::f16;
use hf_hub::api::sync::{Api, ApiRepo};
use ndarray::Array2;
use safetensors::{tensor::Dtype, SafeTensors};
use serde_json::Value;
use std::{
    env, fs,
    path::{Path, PathBuf},
};
use tokenizers::Tokenizer;

/// Which embedding-tensor naming convention a model uses.
#[derive(Debug, Clone, Copy)]
enum ModelLayout {
    Native,
    SentenceTransformers,
}

impl ModelLayout {
    fn embedding_key(self) -> &'static str {
        match self {
            Self::Native => "embeddings",
            Self::SentenceTransformers => "embedding.weight",
        }
    }
}

struct ResolvedPaths {
    tokenizer_path: PathBuf,
    model_path: PathBuf,
    config_path: PathBuf,
    modules_path: Option<PathBuf>,
    layout: ModelLayout,
}

struct Layout {
    config_file: &'static str,
    tokenizer_file: &'static str,
    model_file: &'static str,
    /// Config lives one level above the model files (layout 4: caller pointed at model dir).
    config_in_parent: bool,
    /// Skip when `config_sentence_transformers.json` is present alongside `config.json`.
    skip_if_st_config: bool,
    layout: ModelLayout,
}

static LAYOUTS: &[Layout] = &[
    // 1. Native model2vec
    Layout {
        config_file: "config.json",
        tokenizer_file: "tokenizer.json",
        model_file: "model.safetensors",
        config_in_parent: false,
        skip_if_st_config: true,
        layout: ModelLayout::Native,
    },
    // 2. Sentence Transformers root layout
    Layout {
        config_file: "config_sentence_transformers.json",
        tokenizer_file: "tokenizer.json",
        model_file: "model.safetensors",
        config_in_parent: false,
        skip_if_st_config: false,
        layout: ModelLayout::SentenceTransformers,
    },
    // 3. Sentence Transformers 0_StaticEmbedding subfolder
    Layout {
        config_file: "config_sentence_transformers.json",
        tokenizer_file: "0_StaticEmbedding/tokenizer.json",
        model_file: "0_StaticEmbedding/model.safetensors",
        config_in_parent: false,
        skip_if_st_config: false,
        layout: ModelLayout::SentenceTransformers,
    },
    // 4. Config-in-parent (caller passed subfolder pointing directly at model files)
    Layout {
        config_file: "config_sentence_transformers.json",
        tokenizer_file: "tokenizer.json",
        model_file: "model.safetensors",
        config_in_parent: true,
        skip_if_st_config: false,
        layout: ModelLayout::SentenceTransformers,
    },
];

fn resolve_local(folder: &Path) -> Option<ResolvedPaths> {
    for spec in LAYOUTS {
        let config_base = if spec.config_in_parent {
            folder.parent()?
        } else {
            folder
        };
        let config_path = config_base.join(spec.config_file);
        if !config_path.exists() {
            continue;
        }
        if spec.skip_if_st_config && folder.join("config_sentence_transformers.json").exists() {
            continue;
        }
        let tokenizer_path = folder.join(spec.tokenizer_file);
        let model_path = folder.join(spec.model_file);
        if !tokenizer_path.exists() || !model_path.exists() {
            continue;
        }
        return Some(ResolvedPaths {
            tokenizer_path,
            model_path,
            config_path,
            modules_path: None,
            layout: spec.layout,
        });
    }
    None
}

fn resolve_hub(repo: &ApiRepo, prefix: &str) -> Result<ResolvedPaths> {
    for spec in LAYOUTS {
        let config_prefix: String = if spec.config_in_parent {
            parent_of_prefix(prefix)
        } else {
            prefix.to_string()
        };
        let Ok(config_path) = repo.get(&format!("{config_prefix}{}", spec.config_file)) else {
            continue;
        };
        if spec.skip_if_st_config
            && repo
                .get(&format!("{config_prefix}config_sentence_transformers.json"))
                .is_ok()
        {
            continue;
        }
        let Ok(tokenizer_path) = repo.get(&format!("{prefix}{}", spec.tokenizer_file)) else {
            continue;
        };
        let Ok(model_path) = repo.get(&format!("{prefix}{}", spec.model_file)) else {
            continue;
        };
        let modules_path = repo.get(&format!("{config_prefix}modules.json")).ok();
        return Ok(ResolvedPaths {
            tokenizer_path,
            model_path,
            config_path,
            modules_path,
            layout: spec.layout,
        });
    }
    Err(anyhow!(
        "no valid model layout found. Tried config files: {}",
        LAYOUTS.iter().map(|s| s.config_file).collect::<Vec<_>>().join(", ")
    ))
}

fn parent_of_prefix(prefix: &str) -> String {
    let trimmed = prefix.trim_end_matches('/');
    match Path::new(trimmed).parent() {
        Some(p) if !p.as_os_str().is_empty() => format!("{}/", p.display()),
        _ => String::new(),
    }
}

fn read_config_normalize(config_path: &Path) -> Option<bool> {
    let f = std::fs::File::open(config_path).ok()?;
    let cfg: Value = serde_json::from_reader(f).ok()?;
    cfg.get("normalize").and_then(Value::as_bool)
}

fn has_normalize_module(modules_path: &Path) -> Option<bool> {
    let f = std::fs::File::open(modules_path).ok()?;
    let Value::Array(modules) = serde_json::from_reader::<_, Value>(f).ok()? else {
        return None;
    };
    Some(modules.iter().any(|module| {
        module
            .get("type")
            .and_then(Value::as_str)
            .map(|t| t.contains("Normalize"))
            .unwrap_or(false)
    }))
}

fn read_normalize(config_path: &Path, explicit_modules: Option<&Path>) -> bool {
    if let Some(v) = read_config_normalize(config_path) {
        return v;
    }
    let derived = config_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join("modules.json");
    let modules_path = explicit_modules.unwrap_or(&derived);
    has_normalize_module(modules_path).unwrap_or(true)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

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
    /// Supports four layouts (tried in order):
    /// - **model2vec**: `config.json` + `model.safetensors` + `tokenizer.json`
    /// - **sentence-transformers root**: `config_sentence_transformers.json` + model files at same level
    /// - **0_StaticEmbedding**: `config_sentence_transformers.json` at root, model files under `0_StaticEmbedding/`
    /// - **config-in-parent**: model files in current dir, `config_sentence_transformers.json` one level up
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
        if let Some(tok) = token {
            env::set_var("HF_HUB_TOKEN", tok);
        }

        let base = repo_or_path.as_ref();
        let ResolvedPaths {
            tokenizer_path: tok_path,
            model_path: mdl_path,
            config_path: cfg_path,
            modules_path: mod_path,
            layout,
        } = if base.exists() {
            let folder = subfolder.map(|s| base.join(s)).unwrap_or_else(|| base.to_path_buf());
            resolve_local(&folder).ok_or_else(|| {
                anyhow!(
                    "no valid model layout found in {folder:?}. \
                     Tried: model2vec (config.json), sentence-transformers \
                     (config_sentence_transformers.json), and 0_StaticEmbedding subfolder."
                )
            })?
        } else {
            let api = Api::new().context("hf-hub API init failed")?;
            let repo = api.model(base.to_string_lossy().into_owned());
            let prefix = subfolder.map(|s| format!("{s}/")).unwrap_or_default();
            resolve_hub(&repo, &prefix)
                .with_context(|| format!("could not load '{}' from HuggingFace Hub", base.display()))?
        };

        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;

        let mut lens: Vec<usize> = tokenizer.get_vocab(false).keys().map(|tk| tk.len()).collect();
        lens.sort_unstable();
        let median_token_length = lens.get(lens.len() / 2).copied().unwrap_or(1);

        let normalize = normalize.unwrap_or_else(|| read_normalize(&cfg_path, mod_path.as_deref()));

        let spec_json = tokenizer
            .to_string(false)
            .map_err(|e| anyhow!("tokenizer -> JSON failed: {e}"))?;
        let spec: Value = serde_json::from_str(&spec_json)?;
        // If no unk token is defined, don't filter anything.
        // If one is defined but absent from the vocab, fall back to id 0.
        let unk_token_id = spec
            .get("model")
            .and_then(|m| m.get("unk_token"))
            .and_then(Value::as_str)
            .map(|unk| tokenizer.token_to_id(unk).map(|id| id as usize).unwrap_or(0));

        let model_bytes = fs::read(&mdl_path).context("failed to read model.safetensors")?;
        let safet = SafeTensors::deserialize(&model_bytes).context("failed to parse safetensors")?;
        let emb_key = layout.embedding_key();
        let tensor = safet
            .tensor(emb_key)
            .or_else(|_| safet.tensor("embeddings"))
            .or_else(|_| safet.tensor("embedding.weight"))
            .or_else(|_| safet.tensor("0"))
            .with_context(|| {
                format!("embedding tensor not found (tried '{emb_key}', 'embeddings', 'embedding.weight', '0')")
            })?;

        let [rows, cols]: [usize; 2] = tensor.shape().try_into().context("embedding tensor is not 2-D")?;
        let raw = tensor.data();
        let floats: Vec<f32> = match tensor.dtype() {
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

    /// Char-level truncation to max_tokens × median_token_length
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
        for batch in sentences.chunks(batch_size) {
            let truncated: Vec<&str> = batch
                .iter()
                .map(|text| {
                    max_length
                        .map(|max_tok| Self::truncate_str(text, max_tok, self.median_token_length))
                        .unwrap_or(text.as_str())
                })
                .collect();
            let encodings = self
                .tokenizer
                .encode_batch_fast::<String>(
                    truncated.into_iter().map(Into::into).collect(),
                    /* add_special_tokens = */ false,
                )
                .expect("tokenization failed");
            for encoding in encodings {
                let mut token_ids = encoding.get_ids().to_vec();
                if let Some(unk_id) = self.unk_token_id {
                    token_ids.retain(|&id| id as usize != unk_id);
                }
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

    /// Encode a single sentence into a vector.
    pub fn encode_single(&self, sentence: &str) -> Vec<f32> {
        self.encode(&[sentence.to_string()])
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    /// Mean-pool a token-ID list into a single vector.
    fn pool_ids(&self, ids: Vec<u32>) -> Vec<f32> {
        let dim = self.embeddings.ncols();
        let mut sum = vec![0.0_f32; dim];
        let mut cnt = 0usize;
        for &id in &ids {
            let tok = id as usize;
            let row_idx = self
                .token_mapping
                .as_ref()
                .and_then(|m| m.get(tok))
                .copied()
                .unwrap_or(tok);
            let scale = self.weights.as_ref().and_then(|w| w.get(tok)).copied().unwrap_or(1.0);
            let row = self.embeddings.row(row_idx);
            for (s, &v) in sum.iter_mut().zip(row.iter()) {
                *s += v * scale;
            }
            cnt += 1;
        }
        let denom = cnt.max(1) as f32;
        for x in &mut sum {
            *x /= denom;
        }
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
    fn test_parent_of_prefix() {
        let cases = [
            ("0_StaticEmbedding/", ""),
            ("some/path/0_StaticEmbedding/", "some/path/"),
            ("models/", ""),
            ("a/b/", "a/"),
        ];
        for (prefix, expected) in cases {
            assert_eq!(parent_of_prefix(prefix), expected, "parent_of_prefix({prefix:?})");
        }
    }
}
