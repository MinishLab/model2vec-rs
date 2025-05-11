use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use safetensors::{SafeTensors, tensor::Dtype};
use half::f16;
use ndarray::Array2;
use std::fs::read;
use std::path::Path;
use anyhow::{Result, Context, anyhow};
use serde_json::Value;

/// Static embedding model for Model2Vec
pub struct StaticModel {
    tokenizer: Tokenizer,
    embeddings: Array2<f32>,
    normalize: bool,
}

impl StaticModel {
    /// Load a Model2Vec model from a local folder or the HF Hub.
    ///
    /// # Arguments
    /// * `repo_or_path` - HF repo ID or local filesystem path
    /// * `subfolder` - optional subdirectory inside the repo or folder
    pub fn from_pretrained(repo_or_path: &str, subfolder: Option<&str>) -> Result<Self> {
        // Determine file paths
        let (tok_path, mdl_path, cfg_path) = {
            let base = Path::new(repo_or_path);
            if base.exists() {
                // Local path
                let folder = subfolder.map(|s| base.join(s)).unwrap_or_else(|| base.to_path_buf());
                let t = folder.join("tokenizer.json");
                let m = folder.join("model.safetensors");
                let c = folder.join("config.json");
                if !t.exists() || !m.exists() || !c.exists() {
                    return Err(anyhow!("Local path {:?} missing tokenizer/model/config files", folder));
                }
                (t, m, c)
            } else {
                // HF Hub path
                let api = Api::new().context("Failed to initialize HF Hub API")?;
                let repo = api.model(repo_or_path.to_string());
                let prefix = subfolder.map(|s| format!("{}/", s)).unwrap_or_default();
                let t = repo.get(&format!("{}tokenizer.json", prefix)).context("Failed to download tokenizer.json")?;
                let m = repo.get(&format!("{}model.safetensors", prefix)).context("Failed to download model.safetensors")?;
                let c = repo.get(&format!("{}config.json", prefix)).context("Failed to download config.json")?;
                (t.into(), m.into(), c.into())
            }
        };

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Read safetensors file
        let bytes = read(&mdl_path).context("Failed to read model.safetensors")?;
        let safet = SafeTensors::deserialize(&bytes).context("Failed to parse safetensors")?;
        let tensor = safet.tensor("embeddings").or_else(|_| safet.tensor("0")).context("Embedding tensor not found")?;
        let shape = (tensor.shape()[0] as usize, tensor.shape()[1] as usize);
        let raw = tensor.data();
        let dtype = tensor.dtype();

        // Read config.json for normalization flag
        let cfg_bytes = read(&cfg_path).context("Failed to read config.json")?;
        let cfg: Value = serde_json::from_slice(&cfg_bytes).context("Failed to parse config.json")?;
        let normalize = cfg.get("normalize").and_then(Value::as_bool).unwrap_or(true);

        // Decode raw bytes into Vec<f32> based on dtype
        let floats: Vec<f32> = match dtype {
            Dtype::F32 => raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            Dtype::F16 => raw.chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            Dtype::I8  => raw.iter()
                .map(|&b| (b as i8) as f32)
                .collect(),
            other => return Err(anyhow!("Unsupported tensor dtype: {:?}", other)),
        };

        // Construct ndarray
        let embeddings = Array2::from_shape_vec(shape, floats).context("Failed to create embeddings array")?;

        Ok(Self { tokenizer, embeddings, normalize })
    }

    /// Tokenize input texts into token ID sequences
    pub fn tokenize(&self, texts: &[String]) -> Vec<Vec<u32>> {
        texts.iter().map(|text| {
            let enc = self.tokenizer.encode(text.as_str(), false).expect("Tokenization failed");
            enc.get_ids().to_vec()
        }).collect()
    }

    /// Encode texts into embeddings via mean-pooling and optional L2-normalization
    pub fn encode(&self, texts: &[String]) -> Vec<Vec<f32>> {
        texts.iter().map(|text| {
            let enc = self.tokenizer.encode(text.as_str(), false).expect("Tokenization failed");
            let ids = enc.get_ids();
            let mut sum = vec![0.0f32; self.embeddings.ncols()];
            for &id in ids {
                let row = self.embeddings.row(id as usize);
                for (i, &v) in row.iter().enumerate() {
                    sum[i] += v;
                }
            }
            let count = ids.len().max(1) as f32;
            sum.iter_mut().for_each(|v| *v /= count);
            if self.normalize {
                let norm = sum.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
                sum.iter_mut().for_each(|v| *v /= norm);
            }
            sum
        }).collect()
    }
}
