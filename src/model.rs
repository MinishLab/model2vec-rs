use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use safetensors::{SafeTensors, tensor::Dtype};
use half::f16;
use ndarray::Array2;
use std::{env, fs, path::Path};
use anyhow::{Context, Result, anyhow};
use serde_json::Value;

/// Static embedding model for Model2Vec
pub struct StaticModel {
    tokenizer: Tokenizer,
    embeddings: Array2<f32>,
    normalize: bool,
    median_token_length: usize,
}

impl StaticModel {
    /// Load a Model2Vec model from a local folder or the HF Hub.
    pub fn from_pretrained(
        repo_or_path: &str,
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
            let base = Path::new(repo_or_path);
            if base.exists() {
                let folder = subfolder.map(|s| base.join(s)).unwrap_or_else(|| base.to_path_buf());
                let t = folder.join("tokenizer.json");
                let m = folder.join("model.safetensors");
                let c = folder.join("config.json");
                if !t.exists() || !m.exists() || !c.exists() {
                    return Err(anyhow!("Local path {:?} missing tokenizer/model/config", folder));
                }
                (t, m, c)
            } else {
                let api = Api::new().context("HF Hub API init failed")?;
                let repo = api.model(repo_or_path.to_string());
                let prefix = subfolder.map(|s| format!("{}/", s)).unwrap_or_default();
                let t = repo
                    .get(&format!("{}tokenizer.json", prefix))
                    .context("Failed to download tokenizer.json")?;
                let m = repo
                    .get(&format!("{}model.safetensors", prefix))
                    .context("Failed to download model.safetensors")?;
                let c = repo
                    .get(&format!("{}config.json", prefix))
                    .context("Failed to download config.json")?;
                (t.into(), m.into(), c.into())
            }
        };

        // Load the tokenizer
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Median-token-length hack for pre-truncation
        let mut lens: Vec<usize> = tokenizer
            .get_vocab(false)
            .keys()
            .map(|tk| tk.len())
            .collect();
        lens.sort_unstable();
        let median_token_length = *lens.get(lens.len() / 2).unwrap_or(&1);

        // Read normalize default from config.json
        let cfg_bytes = fs::read(&cfg_path).context("Failed to read config.json")?;
        let cfg: Value = serde_json::from_slice(&cfg_bytes).context("Failed to parse config.json")?;
        let cfg_norm = cfg.get("normalize").and_then(Value::as_bool).unwrap_or(true);
        let normalize = normalize.unwrap_or(cfg_norm);

        // Load the safetensors
        let model_bytes = fs::read(&mdl_path).context("Failed to read model.safetensors")?;
        let safet  = SafeTensors::deserialize(&model_bytes).context("Failed to parse safetensors")?;
        let tensor = safet
            .tensor("embeddings")
            .or_else(|_| safet.tensor("0"))
            .context("No 'embeddings' tensor found")?;
        let (rows, cols) = (tensor.shape()[0] as usize, tensor.shape()[1] as usize);
        let raw   = tensor.data();
        let dtype = tensor.dtype();

        // Decode into f32
        let floats: Vec<f32> = match dtype {
            Dtype::F32 => raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect(),
            Dtype::F16 => raw.chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0],b[1]]).to_f32()).collect(),
            Dtype::I8  => raw.iter().map(|&b| b as i8 as f32).collect(),
            other      => return Err(anyhow!("Unsupported tensor dtype: {:?}", other)),
        };
        let embeddings = Array2::from_shape_vec((rows, cols), floats)
            .context("Failed to build embeddings array")?;

        Ok(Self {
            tokenizer,
            embeddings,
            normalize,
            median_token_length,
        })
    }

    /// Char-level truncation to max_tokens * median_token_length
    fn truncate_str<'a>(s: &'a str, max_tokens: usize, median_len: usize) -> &'a str {
        let max_chars = max_tokens.saturating_mul(median_len);
        // if <= max_chars characters, return whole string
        if s.chars().count() <= max_chars {
            return s;
        }
        // otherwise find the byte index of the (max_chars)th char and cut there
        match s.char_indices().nth(max_chars) {
            Some((byte_idx, _)) => &s[..byte_idx],
            None => s,
        }
    }

    /// Encode texts into embeddings.
    ///
    /// # Arguments
    /// * `texts` - slice of input strings
    /// * `max_length` - max tokens per text (truncation)
    /// * `batch_size` - number of texts per batch
    pub fn encode_with_args(
        &self,
        texts: &[String],
        max_length: Option<usize>,
        batch_size: usize,
    ) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(batch_size) {
            // Truncate the input strings to max_length * median_token_length
            let slices: Vec<&str> = chunk.iter()
                .map(|t| {
                    if let Some(mx) = max_length {
                        Self::truncate_str(t, mx, self.median_token_length)
                    } else {
                        t.as_str()
                    }
                })
                .collect();

            // Tokenize the batch
            let encs = self
                .tokenizer
                .encode_batch_fast::<String>(
                    slices.into_iter().map(|s| s.into()).collect(),
                    false,
                )
                .expect("Tokenization failed");

            // Encode the token IDs into embeddings
            for enc in encs {
                let mut ids = enc.get_ids().to_vec();
                if let Some(mx) = max_length {
                    ids.truncate(mx);
                }
                out.push(self.pool_ids(ids));
            }
        }

        out
    }

    /// Default encode: `max_length=512`, `batch_size=1024`
    pub fn encode(&self, texts: &[String]) -> Vec<Vec<f32>> {
        self.encode_with_args(texts, Some(512), 1024)
    }

    /// Mean-pool a single token-ID list into a vector
    fn pool_ids(&self, ids: Vec<u32>) -> Vec<f32> {
        let mut sum = vec![0.0; self.embeddings.ncols()];
        for &id in &ids {
            let row = self.embeddings.row(id as usize);
            for (i, &v) in row.iter().enumerate() {
                sum[i] += v;
            }
        }
        let cnt = ids.len().max(1) as f32;
        sum.iter_mut().for_each(|x| *x /= cnt);
        if self.normalize {
            let norm = sum.iter().map(|&v| v*v).sum::<f32>().sqrt().max(1e-12);
            sum.iter_mut().for_each(|x| *x /= norm);
        }
        sum
    }
}
