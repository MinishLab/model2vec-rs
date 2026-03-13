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

/// Check whether `config_base/config_file`, `model_base/tokenizer.json`, and
/// `model_base/model.safetensors` all exist, and if so return a `ResolvedPaths`.
fn local_probe(config_base: &Path, model_base: &Path, config_file: &str, layout: ModelLayout) -> Option<ResolvedPaths> {
    let config_path = config_base.join(config_file);
    let tokenizer_path = model_base.join("tokenizer.json");
    let model_path = model_base.join("model.safetensors");
    (config_path.exists() && tokenizer_path.exists() && model_path.exists()).then_some(ResolvedPaths {
        config_path,
        tokenizer_path,
        model_path,
        modules_path: None,
        layout,
    })
}

/// Fetch `config_prefix/config_file`, `model_prefix/tokenizer.json`, and
/// `model_prefix/model.safetensors` from the Hub; return `None` if any is missing.
fn hub_probe(
    repo: &ApiRepo,
    config_prefix: &str,
    model_prefix: &str,
    config_file: &str,
    layout: ModelLayout,
) -> Option<ResolvedPaths> {
    let config_path = repo.get(&format!("{config_prefix}{config_file}")).ok()?;
    let tokenizer_path = repo.get(&format!("{model_prefix}tokenizer.json")).ok()?;
    let model_path = repo.get(&format!("{model_prefix}model.safetensors")).ok()?;
    let modules_path = repo.get(&format!("{config_prefix}modules.json")).ok();
    Some(ResolvedPaths {
        config_path,
        tokenizer_path,
        model_path,
        modules_path,
        layout,
    })
}

fn resolve_local(folder: &Path) -> Option<ResolvedPaths> {
    // Native model2vec — skip when a sentence-transformers config is also present.
    if !folder.join("config_sentence_transformers.json").exists() {
        if let r @ Some(_) = local_probe(folder, folder, "config.json", ModelLayout::Native) {
            return r;
        }
    }
    // Sentence Transformers root layout.
    if let r @ Some(_) = local_probe(
        folder,
        folder,
        "config_sentence_transformers.json",
        ModelLayout::SentenceTransformers,
    ) {
        return r;
    }
    // Sentence Transformers with model files in 0_StaticEmbedding/.
    let sub = folder.join("0_StaticEmbedding");
    if let r @ Some(_) = local_probe(
        folder,
        &sub,
        "config_sentence_transformers.json",
        ModelLayout::SentenceTransformers,
    ) {
        return r;
    }
    // Config lives one level up (caller pointed directly at the model-files directory).
    let parent = folder.parent()?;
    local_probe(
        parent,
        folder,
        "config_sentence_transformers.json",
        ModelLayout::SentenceTransformers,
    )
}

fn resolve_hub(repo: &ApiRepo, prefix: &str) -> Result<ResolvedPaths> {
    // Native model2vec — skip when a sentence-transformers config is also present.
    if repo.get(&format!("{prefix}config_sentence_transformers.json")).is_err() {
        if let Some(r) = hub_probe(repo, prefix, prefix, "config.json", ModelLayout::Native) {
            return Ok(r);
        }
    }
    // Sentence Transformers root layout.
    if let Some(r) = hub_probe(
        repo,
        prefix,
        prefix,
        "config_sentence_transformers.json",
        ModelLayout::SentenceTransformers,
    ) {
        return Ok(r);
    }
    // Sentence Transformers with model files in 0_StaticEmbedding/.
    let sub_prefix = format!("{prefix}0_StaticEmbedding/");
    if let Some(r) = hub_probe(
        repo,
        prefix,
        &sub_prefix,
        "config_sentence_transformers.json",
        ModelLayout::SentenceTransformers,
    ) {
        return Ok(r);
    }
    // Config lives one level up.
    let trimmed = prefix.trim_end_matches('/');
    let parent = match Path::new(trimmed).parent() {
        Some(p) if !p.as_os_str().is_empty() => format!("{}/", p.display()),
        _ => String::new(),
    };
    hub_probe(
        repo,
        &parent,
        prefix,
        "config_sentence_transformers.json",
        ModelLayout::SentenceTransformers,
    )
    .ok_or_else(|| anyhow!("no valid model layout found in '{prefix}'"))
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

        let normalize = normalize.unwrap_or_else(|| {
            // 1. explicit key in config file
            if let Some(v) = std::fs::File::open(&cfg_path)
                .ok()
                .and_then(|f| serde_json::from_reader::<_, Value>(f).ok())
                .and_then(|cfg| cfg.get("normalize").and_then(Value::as_bool))
            {
                return v;
            }
            // 2. Normalize stage in modules.json; default true
            let derived = cfg_path.parent().unwrap_or_else(|| Path::new("")).join("modules.json");
            let modules_path = mod_path.as_deref().unwrap_or(&derived);
            let Ok(f) = std::fs::File::open(modules_path) else {
                return true;
            };
            let Ok(Value::Array(modules)) = serde_json::from_reader::<_, Value>(f) else {
                return true;
            };
            modules.iter().any(|m| {
                m.get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|t| t.contains("Normalize"))
            })
        });

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
