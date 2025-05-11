use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use safetensors::{SafeTensors, tensor::Dtype};
use half::f16;
use ndarray::Array2;
use rayon::prelude::*;
use std::{fs::read, path::Path, env};
use anyhow::{Result, Context, anyhow};
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

        // Determine file paths
        let (tok_path, mdl_path, cfg_path) = {
            let base = Path::new(repo_or_path);
            if base.exists() {
                let folder = subfolder.map(|s| base.join(s)).unwrap_or_else(|| base.to_path_buf());
                let t = folder.join("tokenizer.json");
                let m = folder.join("model.safetensors");
                let c = folder.join("config.json");
                if !t.exists() || !m.exists() || !c.exists() {
                    return Err(anyhow!("Local path {:?} missing files", folder));
                }
                (t, m, c)
            } else {
                let api = Api::new().context("HF Hub API init failed")?;
                let repo = api.model(repo_or_path.to_string());
                // note: token not used with sync Api
                let prefix = subfolder.map(|s| format!("{}/", s)).unwrap_or_default();
                let t = repo.get(&format!("{}tokenizer.json", prefix))
                    .context("Download tokenizer.json failed")?;
                let m = repo.get(&format!("{}model.safetensors", prefix))
                    .context("Download model.safetensors failed")?;
                let c = repo.get(&format!("{}config.json", prefix))
                    .context("Download config.json failed")?;
                (t.into(), m.into(), c.into())
            }
        };

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow!("Tokenizer load error: {}", e))?;

        // Median token length for char-level truncation
        let mut lengths: Vec<usize> = tokenizer.get_vocab(false)
            .keys().map(|tk| tk.len()).collect();
        lengths.sort_unstable();
        let median_token_length = *lengths.get(lengths.len() / 2).unwrap_or(&1);

        // Read config.json for default normalize
        let cfg: Value = serde_json::from_slice(&read(&cfg_path)?)
            .context("Parse config.json failed")?;
        let config_norm = cfg.get("normalize").and_then(Value::as_bool).unwrap_or(true);
        let normalize = normalize.unwrap_or(config_norm);

        // Read safetensors
        let bytes = read(&mdl_path).context("Read safetensors failed")?;
        let safet = SafeTensors::deserialize(&bytes).context("Parse safetensors failed")?;
        let tensor = safet.tensor("embeddings").or_else(|_| safet.tensor("0"))
            .context("No 'embeddings' tensor")?;
        let shape = (tensor.shape()[0] as usize, tensor.shape()[1] as usize);
        let raw = tensor.data();
        let dtype = tensor.dtype();

        // Decode raw data to f32
        let floats: Vec<f32> = match dtype {
            Dtype::F32 => raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect(),
            Dtype::F16 => raw.chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0],b[1]]).to_f32()).collect(),
            Dtype::I8  => raw.iter().map(|&b| b as i8 as f32).collect(),
            other      => return Err(anyhow!("Unsupported dtype: {:?}", other)),
        };
        let embeddings = Array2::from_shape_vec(shape, floats)
            .context("Array shape error")?;

        Ok(Self { tokenizer, embeddings, normalize, median_token_length })
    }

    /// Tokenize input texts into token ID sequences with optional truncation.
    pub fn tokenize(&self, texts: &[String], max_length: Option<usize>) -> Vec<Vec<u32>> {
        let prepared: Vec<String> = texts.iter().map(|t| {
            if let Some(max) = max_length {
                t.chars().take(max.saturating_mul(self.median_token_length)).collect()
            } else { t.clone() }
        }).collect();
        let encs = self.tokenizer.encode_batch(prepared, false).expect("Tokenization failed");
        encs.into_iter().map(|enc| {
            let mut ids = enc.get_ids().to_vec(); if let Some(max) = max_length { ids.truncate(max); } ids
        }).collect()
    }

    /// Encode texts into embeddings.
    ///
    /// # Arguments
    /// * `texts` - slice of input strings
    /// * `show_progress` - whether to print batch progress
    /// * `max_length` - max tokens per text (truncation)
    /// * `batch_size` - number of texts per batch
    /// * `use_parallel` - use Rayon parallelism
    /// * `parallel_threshold` - minimum texts to enable parallelism
    pub fn encode_with_args(
        &self,
        texts: &[String],
        show_progress: bool,
        max_length: Option<usize>,
        batch_size: usize,
        use_multiprocessing: bool,
        multiprocessing_threshold: usize,
    ) -> Vec<Vec<f32>> {
        let total = texts.len();
        let num_batches = (total + batch_size - 1) / batch_size;
        let iter = texts.chunks(batch_size);

        if use_multiprocessing && total > multiprocessing_threshold {
            // disable tokenizer internal parallel
            env::set_var("TOKENIZERS_PARALLELISM", "false");
            iter
                .enumerate()
                .flat_map(|(b, chunk)| {
                    if show_progress { eprintln!("Batch {}/{}", b+1, num_batches); }
                    self.tokenize(chunk, max_length)
                        .into_par_iter()
                        .map(|ids| self.pool_ids(ids))
                        .collect::<Vec<_>>()
                })
                .collect()
        } else {
            let mut out = Vec::with_capacity(total);
            for (b, chunk) in iter.enumerate() {
                if show_progress { eprintln!("Batch {}/{}", b+1, num_batches); }
                for ids in self.tokenize(chunk, max_length) {
                    out.push(self.pool_ids(ids));
                }
            }
            out
        }
    }

    /// Default encode: no progress, max_length=512, batch_size=1024, no parallel.
    pub fn encode(&self, texts: &[String]) -> Vec<Vec<f32>> {
        self.encode_with_args(texts, false, Some(512), 1024, true, 10_000)
    }

    /// Mean-pool one ID list to embedding
    fn pool_ids(&self, ids: Vec<u32>) -> Vec<f32> {
        let mut sum = vec![0.0; self.embeddings.ncols()];
        for &id in &ids {
            let row = self.embeddings.row(id as usize);
            for (i, &v) in row.iter().enumerate() { sum[i] += v; }
        }
        let cnt = ids.len().max(1) as f32;
        sum.iter_mut().for_each(|v| *v /= cnt);
        if self.normalize {
            let norm = sum.iter().map(|&x| x*x).sum::<f32>().sqrt().max(1e-12);
            sum.iter_mut().for_each(|v| *v /= norm);
        }
        sum
    }
}

