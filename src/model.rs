use anyhow::{anyhow, Context, Result};
use half::f16;
use hf_hub::api::sync::Api;
use ndarray::Array2;
use safetensors::{tensor::Dtype, SafeTensors};
use serde_json::Value;
use std::{env, fs, path::Path};
use tokenizers::Tokenizer;

/// Static embedding model for Model2Vec
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
    /// * `normalize` - Optional flag to normalize embeddings (default from config.json).
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

        // Locate tokenizer.json, model.safetensors, config.json
        let (tok_path, mdl_path, cfg_path) = {
            let base = repo_or_path.as_ref();
            if base.exists() {
                let folder = subfolder.map(|s| base.join(s)).unwrap_or_else(|| base.to_path_buf());
                let t = folder.join("tokenizer.json");
                let m = folder.join("model.safetensors");
                let c = folder.join("config.json");
                if !t.exists() || !m.exists() || !c.exists() {
                    return Err(anyhow!("local path {folder:?} missing tokenizer / model / config"));
                }
                (t, m, c)
            } else {
                let api = Api::new().context("hf-hub API init failed")?;
                let repo = api.model(repo_or_path.as_ref().to_string_lossy().into_owned());
                let prefix = subfolder.map(|s| format!("{}/", s)).unwrap_or_default();
                let t = repo.get(&format!("{prefix}tokenizer.json"))?;
                let m = repo.get(&format!("{prefix}model.safetensors"))?;
                let c = repo.get(&format!("{prefix}config.json"))?;
                (t, m, c)
            }
        };

        // Load the tokenizer
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;

        // Median-token-length hack for pre-truncation
        let mut lens: Vec<usize> = tokenizer.get_vocab(false).keys().map(|tk| tk.len()).collect();
        lens.sort_unstable();
        let median_token_length = lens.get(lens.len() / 2).copied().unwrap_or(1);

        // Read normalize default from config.json
        let cfg_file = std::fs::File::open(&cfg_path).context("failed to read config.json")?;
        let cfg: Value = serde_json::from_reader(&cfg_file).context("failed to parse config.json")?;
        let cfg_norm = cfg.get("normalize").and_then(Value::as_bool).unwrap_or(true);
        let normalize = normalize.unwrap_or(cfg_norm);

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
        let unk_token_id = tokenizer
            .token_to_id(unk_token)
            .ok_or_else(|| anyhow!("tokenizer claims unk_token='{unk_token}' but it isn't in the vocab"))?
            as usize;

        // Load the safetensors
        let model_bytes = fs::read(&mdl_path).context("failed to read model.safetensors")?;
        let safet = SafeTensors::deserialize(&model_bytes).context("failed to parse safetensors")?;
        let tensor = safet
            .tensor("embeddings")
            .or_else(|_| safet.tensor("0"))
            .context("embeddings tensor not found")?;

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
                    Dtype::F64 => raw.chunks_exact(8)
                        .map(|b| f64::from_le_bytes(b.try_into().unwrap()) as f32)
                        .collect(),
                    Dtype::F32 => raw.chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect(),
                    Dtype::F16 => raw.chunks_exact(2)
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
                let v: Vec<usize> = raw.chunks_exact(4)
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
            unk_token_id: Some(unk_token_id),
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
