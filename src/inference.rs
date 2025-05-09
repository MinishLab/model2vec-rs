use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use safetensors::SafeTensors;
use ndarray::Array2;
use std::fs::read;
use anyhow::{Result, Context, anyhow};
use serde_json::Value;

/// Static embedding model loader and encoder for Model2Vec 
pub struct StaticModel {
    tokenizer: Tokenizer,
    embeddings: Array2<f32>,
    normalize: bool,
}

impl StaticModel {
    /// Load a Model2Vec model from the Hugging Face Hub
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        // Initialize HF Hub API
        let api = Api::new().context("Failed to create HF Hub API")?;
        let repo = api.model(repo_id.to_string());

        // Download and load tokenizer
        let tok_path = repo.get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Download embeddings
        let model_path = repo.get("model.safetensors")
            .context("Failed to download model.safetensors")?;
        let bytes = read(&model_path)
            .context("Failed to read model.safetensors")?;
        let safet = SafeTensors::deserialize(&bytes)
            .context("Failed to parse safetensors")?;
        let tensor = safet.tensor("embeddings")
            .or_else(|_| safet.tensor("0"))
            .context("Embedding tensor not found")?;
        let shape = (tensor.shape()[0] as usize, tensor.shape()[1] as usize);
        let raw = tensor.data();
        // Interpret bytes as little-endian f32
        let floats: Vec<f32> = raw.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let embeddings = Array2::from_shape_vec(shape, floats)
            .context("Failed to create embeddings array")?;

        // Download and parse config
        let cfg_bytes = read(&repo.get("config.json")
            .context("Failed to download config.json")?)
            .context("Failed to read config.json")?;
        let cfg: Value = serde_json::from_slice(&cfg_bytes)
            .context("Failed to parse config.json")?;
        let normalize = cfg.get("normalize").and_then(Value::as_bool).unwrap_or(true);

        Ok(Self { tokenizer, embeddings, normalize })
    }

    /// Encode input texts into embeddings via mean-pooling and optional L2-normalization
    pub fn encode(&self, texts: &[String]) -> Vec<Vec<f32>> {
        texts.iter().map(|text| {
            // Tokenize without special tokens
            let encoding = self.tokenizer.encode(text.as_str(), false)
                .expect("Tokenization failed");
            let ids = encoding.get_ids();

            // Mean-pool token embeddings
            let mut sum = vec![0.0f32; self.embeddings.ncols()];
            for &id in ids {
                let row = self.embeddings.row(id as usize);
                for (i, &v) in row.iter().enumerate() {
                    sum[i] += v;
                }
            }
            let count = ids.len().max(1) as f32;
            for v in &mut sum {
                *v /= count;
            }

            // Optional L2-normalize
            if self.normalize {
                let norm = sum.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
                for v in &mut sum {
                    *v /= norm;
                }
            }
            sum
        }).collect()
    }
}