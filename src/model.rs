use anyhow::{Context, Result, anyhow};
use half::f16;
#[cfg(all(feature = "hf-hub", not(feature = "local-only")))]
use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};
use safetensors::{SafeTensors, tensor::Dtype};
use serde_json::Value;
use std::borrow::Cow;
use std::cell::RefCell;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokenizers::{Tokenizer, models::ModelWrapper, parallelism::MaybeParallelRefIterator};

use crate::fast_unigram::FastUnigram;
use crate::fast_wordpiece::FastWordPiece;

thread_local! {
    static TOKEN_IDS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// Static embedding model for Model2Vec.
///
/// Clones share immutable model state instead of copying the tokenizer and embeddings.
#[derive(Debug, Clone)]
pub struct StaticModel {
    inner: Arc<StaticModelInner>,
}

#[derive(Debug)]
struct StaticModelInner {
    tokenizer: Tokenizer,
    embeddings: Embeddings,
    dim: usize,
    weights: Option<Cow<'static, [f32]>>,
    token_mapping: Option<Cow<'static, [usize]>>,
    normalize: bool,
    median_token_length: usize,
    unk_token_id: Option<usize>,
    /// Custom IDs-only WordPiece tokenizer; `None` for non-WordPiece
    /// pipelines, which fall back to the `tokenizers` crate.
    fast_tok: Option<FastWordPiece>,
    /// Custom IDs-only Unigram model; `None` for non-Unigram pipelines and
    /// for those it does not reproduce, which fall back to the `tokenizers` crate.
    fast_unigram: Option<FastUnigram>,
}

/// Embedding table in its stored element type, row-major with `dim` columns.
#[derive(Debug)]
enum Embeddings {
    F32(Cow<'static, [f32]>),
    I8(Cow<'static, [i8]>),
}

impl Embeddings {
    fn len(&self) -> usize {
        match self {
            Embeddings::F32(table) => table.len(),
            Embeddings::I8(table) => table.len(),
        }
    }
}

#[derive(Debug, Clone)]
struct ModelFiles {
    tokenizer: PathBuf,
    model: PathBuf,
    config: PathBuf,
}

fn layout_candidates<T: Copy>(base: T, subfolder: T, parent: T) -> [(T, T, &'static str); 4] {
    [
        (base, base, "config.json"),
        (base, base, "config_sentence_transformers.json"),
        (base, subfolder, "config_sentence_transformers.json"),
        (parent, base, "config_sentence_transformers.json"),
    ]
}

fn decode_f32(raw: &[u8]) -> Vec<f32> {
    raw.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect()
}

fn decode_f16(raw: &[u8]) -> Vec<f32> {
    raw.as_chunks::<2>()
        .0
        .iter()
        .map(|&b| f16::from_le_bytes(b).to_f32())
        .collect()
}

fn match_local_layout(config_base: &Path, model_base: &Path, config_file: &str) -> Option<ModelFiles> {
    let config = config_base.join(config_file);
    let tokenizer = model_base.join("tokenizer.json");
    let model = model_base.join("model.safetensors");
    (config.exists() && tokenizer.exists() && model.exists()).then_some(ModelFiles {
        tokenizer,
        model,
        config,
    })
}

fn decode_token_mapping(dtype: Dtype, raw: &[u8]) -> Result<Vec<usize>> {
    let mapping = match dtype {
        Dtype::I64 => raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&b| i64::from_le_bytes(b) as usize)
            .collect(),
        Dtype::I32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| i32::from_le_bytes(b) as usize)
            .collect(),
        other => return Err(anyhow!("unsupported mapping dtype: {:?}", other)),
    };

    Ok(mapping)
}

#[cfg(all(feature = "hf-hub", not(feature = "local-only")))]
fn is_not_found(e: &hf_hub::api::sync::ApiError) -> bool {
    use hf_hub::api::sync::ApiError;

    matches!(e, ApiError::RequestError(e) if matches!(e.as_ref(), ureq::Error::Status(404, _)))
}

#[cfg(all(feature = "hf-hub", not(feature = "local-only")))]
fn match_hub_layout(
    repo: &ApiRepo,
    config_prefix: &str,
    model_prefix: &str,
    config_file: &str,
) -> Result<Option<ModelFiles>> {
    let fetch = |path: String| -> Result<Option<PathBuf>> {
        match repo.get(&path) {
            Ok(p) => Ok(Some(p)),
            Err(e) if is_not_found(&e) => Ok(None),
            Err(e) => Err(e.into()),
        }
    };
    let Some(config) = fetch(format!("{config_prefix}{config_file}"))? else {
        return Ok(None);
    };
    let Some(tokenizer) = fetch(format!("{model_prefix}tokenizer.json"))? else {
        return Ok(None);
    };
    let Some(model) = fetch(format!("{model_prefix}model.safetensors"))? else {
        return Ok(None);
    };
    Ok(Some(ModelFiles {
        tokenizer,
        model,
        config,
    }))
}

fn resolve_local_model_files(folder: &Path) -> Option<ModelFiles> {
    let subfolder = folder.join("0_StaticEmbedding");
    layout_candidates(Some(folder), Some(subfolder.as_path()), folder.parent())
        .into_iter()
        .find_map(|(config_base, model_base, config_file)| match_local_layout(config_base?, model_base?, config_file))
}

#[cfg(all(feature = "hf-hub", not(feature = "local-only")))]
fn resolve_hub_model_files(repo: &ApiRepo, prefix: &str) -> Result<ModelFiles> {
    let sub_prefix = format!("{prefix}0_StaticEmbedding/");
    let trimmed = prefix.trim_end_matches('/');
    let parent = match Path::new(trimmed).parent() {
        Some(path) if !path.as_os_str().is_empty() => format!("{}/", path.display()),
        _ => String::new(),
    };

    for (config_prefix, model_prefix, config_file) in layout_candidates(prefix, sub_prefix.as_str(), parent.as_str()) {
        if let Some(files) = match_hub_layout(repo, config_prefix, model_prefix, config_file)? {
            return Ok(files);
        }
    }
    Err(anyhow!("no valid model layout found in '{prefix}'"))
}

impl StaticModel {
    /// Load a Model2Vec model directly from in-memory bytes.
    ///
    /// This path is useful for runtimes that fetch model assets as bytes
    /// rather than reading them from a local filesystem.
    pub fn from_bytes<T, M, C>(
        tokenizer_bytes: T,
        model_bytes: M,
        config_bytes: C,
        normalize: Option<bool>,
    ) -> Result<Self>
    where
        T: AsRef<[u8]>,
        M: AsRef<[u8]>,
        C: AsRef<[u8]>,
    {
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes).map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;

        // Read normalize default from config.json
        let cfg: Value = serde_json::from_slice(config_bytes.as_ref()).context("failed to parse config.json")?;
        let cfg_norm = cfg.get("normalize").and_then(Value::as_bool).unwrap_or(true);
        let normalize = normalize.unwrap_or(cfg_norm);

        // Load the safetensors
        let safet = SafeTensors::deserialize(model_bytes.as_ref()).context("failed to parse safetensors")?;
        let tensor = safet
            .tensor("embeddings")
            .or_else(|_| safet.tensor("0"))
            .or_else(|_| safet.tensor("embedding.weight"))
            .context("embeddings tensor not found")?;

        let [rows, cols]: [usize; 2] = tensor.shape().try_into().context("embedding tensor is not 2-D")?;
        let raw = tensor.data();
        let embeddings = match tensor.dtype() {
            Dtype::F32 => Embeddings::F32(Cow::Owned(decode_f32(raw))),
            Dtype::F16 => Embeddings::F32(Cow::Owned(decode_f16(raw))),
            Dtype::I8 => Embeddings::I8(Cow::Owned(raw.iter().map(|&b| b as i8).collect())),
            other => return Err(anyhow!("unsupported tensor dtype: {other:?}")),
        };

        let weights = match safet.tensor("weights") {
            Ok(t) => {
                let raw = t.data();
                let v: Vec<f32> = match t.dtype() {
                    Dtype::F64 => raw
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|&b| f64::from_le_bytes(b) as f32)
                        .collect(),
                    Dtype::F32 => decode_f32(raw),
                    Dtype::F16 => decode_f16(raw),
                    other => return Err(anyhow!("unsupported weights dtype: {:?}", other)),
                };
                Some(v)
            }
            Err(_) => None,
        };

        let token_mapping = match safet.tensor("mapping") {
            Ok(t) => Some(decode_token_mapping(t.dtype(), t.data())?),
            Err(_) => None,
        };

        Self::from_storage(
            tokenizer,
            embeddings,
            rows,
            cols,
            normalize,
            weights.map(Cow::Owned),
            token_mapping.map(Cow::Owned),
        )
    }

    /// Load a Model2Vec model from a local folder or the HuggingFace Hub.
    ///
    /// # Arguments
    /// * `repo_or_path` - HuggingFace repo ID or local path to the model folder.
    /// * `token` - Optional HuggingFace token for authenticated downloads.
    /// * `normalize` - Optional flag to normalize embeddings (default from the resolved config file).
    /// * `subfolder` - Optional subfolder within the repo or path to look for model files.
    pub fn from_pretrained<P: AsRef<Path>>(
        repo_or_path: P,
        token: Option<&str>,
        normalize: Option<bool>,
        subfolder: Option<&str>,
    ) -> Result<Self> {
        let files = resolve_model_files(repo_or_path, token, subfolder)?;
        let tokenizer_bytes = fs::read(&files.tokenizer).context("failed to read tokenizer.json")?;
        let model_bytes = fs::read(&files.model).context("failed to read model.safetensors")?;
        let config_bytes = fs::read(&files.config).context("failed to read config.json")?;
        Self::from_bytes(tokenizer_bytes, model_bytes, config_bytes, normalize)
    }

    /// Construct from owned data.
    ///
    /// # Arguments
    /// * `tokenizer` - Pre-deserialized tokenizer
    /// * `embeddings` - Owned f32 embedding data
    /// * `rows` - Number of vocabulary entries
    /// * `cols` - Embedding dimension
    /// * `normalize` - Whether to L2-normalize output embeddings
    /// * `weights` - Optional per-token weights for quantized models
    /// * `token_mapping` - Optional token ID mapping for quantized models
    pub fn from_owned(
        tokenizer: Tokenizer,
        embeddings: Vec<f32>,
        rows: usize,
        cols: usize,
        normalize: bool,
        weights: Option<Vec<f32>>,
        token_mapping: Option<Vec<usize>>,
    ) -> Result<Self> {
        Self::from_storage(
            tokenizer,
            Embeddings::F32(Cow::Owned(embeddings)),
            rows,
            cols,
            normalize,
            weights.map(Cow::Owned),
            token_mapping.map(Cow::Owned),
        )
    }

    /// Construct from static slices (zero-copy for embedded binary data).
    ///
    /// # Arguments
    /// * `tokenizer` - Pre-deserialized tokenizer
    /// * `embeddings` - Static f32 embedding data (borrowed, no copy)
    /// * `rows` - Number of vocabulary entries
    /// * `cols` - Embedding dimension
    /// * `normalize` - Whether to L2-normalize output embeddings
    /// * `weights` - Optional static per-token weights for quantized models
    /// * `token_mapping` - Optional static token ID mapping for quantized models
    pub fn from_borrowed(
        tokenizer: Tokenizer,
        embeddings: &'static [f32],
        rows: usize,
        cols: usize,
        normalize: bool,
        weights: Option<&'static [f32]>,
        token_mapping: Option<&'static [usize]>,
    ) -> Result<Self> {
        Self::from_storage(
            tokenizer,
            Embeddings::F32(Cow::Borrowed(embeddings)),
            rows,
            cols,
            normalize,
            weights.map(Cow::Borrowed),
            token_mapping.map(Cow::Borrowed),
        )
    }

    /// Construct from owned int8 data, as written by model2vec's
    /// `quantize_to="int8"`. The embeddings stay int8 in memory.
    ///
    /// # Arguments
    /// * `tokenizer` - Pre-deserialized tokenizer
    /// * `embeddings` - Owned int8 embedding data
    /// * `rows` - Number of vocabulary entries
    /// * `cols` - Embedding dimension
    /// * `normalize` - Whether to L2-normalize output embeddings
    /// * `weights` - Optional per-token weights for quantized models
    /// * `token_mapping` - Optional token ID mapping for quantized models
    pub fn from_owned_i8(
        tokenizer: Tokenizer,
        embeddings: Vec<i8>,
        rows: usize,
        cols: usize,
        normalize: bool,
        weights: Option<Vec<f32>>,
        token_mapping: Option<Vec<usize>>,
    ) -> Result<Self> {
        Self::from_storage(
            tokenizer,
            Embeddings::I8(Cow::Owned(embeddings)),
            rows,
            cols,
            normalize,
            weights.map(Cow::Owned),
            token_mapping.map(Cow::Owned),
        )
    }

    /// Construct from static int8 slices (zero-copy for embedded binary data),
    /// as written by model2vec's `quantize_to="int8"`.
    ///
    /// # Arguments
    /// * `tokenizer` - Pre-deserialized tokenizer
    /// * `embeddings` - Static int8 embedding data (borrowed, no copy)
    /// * `rows` - Number of vocabulary entries
    /// * `cols` - Embedding dimension
    /// * `normalize` - Whether to L2-normalize output embeddings
    /// * `weights` - Optional static per-token weights for quantized models
    /// * `token_mapping` - Optional static token ID mapping for quantized models
    pub fn from_borrowed_i8(
        tokenizer: Tokenizer,
        embeddings: &'static [i8],
        rows: usize,
        cols: usize,
        normalize: bool,
        weights: Option<&'static [f32]>,
        token_mapping: Option<&'static [usize]>,
    ) -> Result<Self> {
        Self::from_storage(
            tokenizer,
            Embeddings::I8(Cow::Borrowed(embeddings)),
            rows,
            cols,
            normalize,
            weights.map(Cow::Borrowed),
            token_mapping.map(Cow::Borrowed),
        )
    }

    fn from_storage(
        tokenizer: Tokenizer,
        embeddings: Embeddings,
        rows: usize,
        cols: usize,
        normalize: bool,
        weights: Option<Cow<'static, [f32]>>,
        token_mapping: Option<Cow<'static, [usize]>>,
    ) -> Result<Self> {
        if embeddings.len() != rows * cols {
            return Err(anyhow!(
                "embeddings length {} != rows {} * cols {}",
                embeddings.len(),
                rows,
                cols
            ));
        }
        let (median_token_length, unk_token_id) = Self::compute_metadata(&tokenizer)?;
        let fast_tok = FastWordPiece::from_tokenizer(&tokenizer);
        let fast_unigram = FastUnigram::from_tokenizer(&tokenizer, unk_token_id);
        Ok(Self {
            inner: Arc::new(StaticModelInner {
                tokenizer,
                embeddings,
                dim: cols,
                weights,
                token_mapping,
                normalize,
                median_token_length,
                unk_token_id,
                fast_tok,
                fast_unigram,
            }),
        })
    }

    /// Compute median token length and unk_token_id from tokenizer.
    fn compute_metadata(tokenizer: &Tokenizer) -> Result<(usize, Option<usize>)> {
        let mut lens: Vec<usize> = tokenizer.get_vocab(false).keys().map(|tk| tk.len()).collect();
        lens.sort_unstable();
        let median_token_length = lens.get(lens.len() / 2).copied().unwrap_or(1);

        let lookup = |tok: &str| {
            tokenizer
                .token_to_id(tok)
                .map(|id| id as usize)
                .ok_or_else(|| anyhow!("unk_token '{tok}' not found in vocabulary"))
        };
        let unk_token_id = match tokenizer.get_model() {
            ModelWrapper::BPE(model) => model.unk_token.as_deref().map(lookup).transpose()?,
            ModelWrapper::WordPiece(model) => Some(lookup(&model.unk_token)?),
            ModelWrapper::WordLevel(model) => Some(lookup(&model.unk_token)?),
            // `tokenizers` keeps Unigram's `unk_id` private, so read it from the serialized model.
            ModelWrapper::Unigram(model) => serde_json::to_value(model)
                .context("failed to serialize unigram model")?
                .get("unk_id")
                .and_then(Value::as_u64)
                .map(|id| id as usize),
        };

        Ok((median_token_length, unk_token_id))
    }

    /// Char-level truncation to max_tokens * median_token_length
    fn truncate_str(s: &str, max_tokens: usize, median_len: usize) -> &str {
        s.char_indices()
            .nth(max_tokens.saturating_mul(median_len))
            .map_or(s, |(byte_idx, _)| &s[..byte_idx])
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
            let pooled: Vec<Vec<f32>> = batch
                .maybe_par_iter()
                .map(|text| self.encode_one(text, max_length))
                .collect();
            embeddings.extend(pooled);
        }
        embeddings
    }

    /// Default encode: `max_length=512`, `batch_size=1024`
    pub fn encode(&self, sentences: &[String]) -> Vec<Vec<f32>> {
        self.encode_with_args(sentences, Some(512), 1024)
    }

    /// Encode a single sentence into a vector.
    pub fn encode_single(&self, sentence: &str) -> Vec<f32> {
        self.encode_one(sentence, Some(512))
    }

    /// Truncate, tokenize and pool one text.
    /// Uses the custom WordPiece tokenizer or Unigram model when the model
    /// matches that pipeline; other pipelines use the `tokenizers` crate.
    fn encode_one(&self, text: &str, max_length: Option<usize>) -> Vec<f32> {
        let model = &self.inner;
        let text = max_length.map_or(text, |max_tok| {
            Self::truncate_str(text, max_tok, model.median_token_length)
        });
        match (&model.fast_tok, &model.fast_unigram) {
            (Some(ft), _) => TOKEN_IDS.with_borrow_mut(|ids| {
                ids.clear();
                ft.encode_ids_into(text, ids);
                self.pool_ids(ids, max_length)
            }),
            (None, Some(fu)) => {
                let ids = fu.encode_ids(&model.tokenizer, text).expect("tokenization failed");
                self.pool_ids(&ids, max_length)
            }
            (None, None) => {
                let encoding = model.tokenizer.encode_fast(text, false).expect("tokenization failed");
                self.pool_ids(encoding.get_ids(), max_length)
            }
        }
    }

    /// Mean-pool the first `max_length` token IDs, skipping the unk token.
    fn pool_ids(&self, ids: &[u32], max_length: Option<usize>) -> Vec<f32> {
        let model = &self.inner;
        let ids = &ids[..max_length.map_or(ids.len(), |max_tok| max_tok.min(ids.len()))];
        let (mut sum, cnt) = match &model.embeddings {
            Embeddings::F32(table) => self.sum_rows(table, ids),
            Embeddings::I8(table) => self.sum_rows(table, ids),
        };
        let denom = cnt.max(1) as f32;
        for x in &mut sum {
            *x /= denom;
        }
        if model.normalize {
            let norm = sum.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-12);
            for x in &mut sum {
                *x /= norm;
            }
        }
        sum
    }

    /// Weighted row sum, using a compile-time dimension when one matches.
    fn sum_rows<T: Copy>(&self, table: &[T], ids: &[u32]) -> (Vec<f32>, usize)
    where
        f32: From<T>,
    {
        match self.inner.dim {
            64 => self.sum_rows_fixed::<T, 64>(table, ids),
            128 => self.sum_rows_fixed::<T, 128>(table, ids),
            256 => self.sum_rows_fixed::<T, 256>(table, ids),
            384 => self.sum_rows_fixed::<T, 384>(table, ids),
            512 => self.sum_rows_fixed::<T, 512>(table, ids),
            768 => self.sum_rows_fixed::<T, 768>(table, ids),
            1024 => self.sum_rows_fixed::<T, 1024>(table, ids),
            _ => self.sum_rows_dyn(table, ids),
        }
    }

    /// Weighted row sum for a compile-time embedding dimension.
    fn sum_rows_fixed<T: Copy, const D: usize>(&self, table: &[T], ids: &[u32]) -> (Vec<f32>, usize)
    where
        f32: From<T>,
    {
        let mut sum = [0.0_f32; D];
        let cnt = self.for_each_row(table, ids, |row, scale| {
            let row: &[T; D] = row.try_into().expect("embedding row has D columns");
            for (s, &v) in sum.iter_mut().zip(row) {
                *s += f32::from(v) * scale;
            }
        });
        (sum.to_vec(), cnt)
    }

    /// Weighted row sum for any embedding dimension.
    fn sum_rows_dyn<T: Copy>(&self, table: &[T], ids: &[u32]) -> (Vec<f32>, usize)
    where
        f32: From<T>,
    {
        let mut sum = vec![0.0_f32; self.inner.dim];
        let cnt = self.for_each_row(table, ids, |row, scale| {
            for (s, &v) in sum.iter_mut().zip(row) {
                *s += f32::from(v) * scale;
            }
        });
        (sum, cnt)
    }

    /// Call `f` with the embedding row and weight of every non-unk token;
    /// returns the number of rows visited.
    #[inline(always)]
    fn for_each_row<T>(&self, table: &[T], ids: &[u32], mut f: impl FnMut(&[T], f32)) -> usize {
        let model = &self.inner;
        let dim = model.dim;
        let mut cnt = 0;
        let tokens = ids
            .iter()
            .map(|&id| id as usize)
            .filter(|&tok| Some(tok) != model.unk_token_id);
        match (&model.token_mapping, &model.weights) {
            (None, None) => {
                for tok in tokens {
                    f(&table[tok * dim..(tok + 1) * dim], 1.0);
                    cnt += 1;
                }
            }
            (mapping, weights) => {
                for tok in tokens {
                    let row_idx = mapping.as_ref().and_then(|m| m.get(tok)).copied().unwrap_or(tok);
                    let scale = weights.as_ref().and_then(|w| w.get(tok)).copied().unwrap_or(1.0);
                    f(&table[row_idx * dim..(row_idx + 1) * dim], scale);
                    cnt += 1;
                }
            }
        }
        cnt
    }
}

fn resolve_model_files<P: AsRef<Path>>(
    repo_or_path: P,
    token: Option<&str>,
    subfolder: Option<&str>,
) -> Result<ModelFiles> {
    #[cfg(any(not(feature = "hf-hub"), feature = "local-only"))]
    let _ = token;

    let base = repo_or_path.as_ref();
    if base.exists() {
        let folder = subfolder.map(|s| base.join(s)).unwrap_or_else(|| base.to_path_buf());
        return resolve_local_model_files(&folder).ok_or_else(|| {
            anyhow!(
                "no valid model layout found in {folder:?}. \
                 Tried: model2vec (config.json), sentence-transformers \
                 (config_sentence_transformers.json), and 0_StaticEmbedding subfolder."
            )
        });
    }

    #[cfg(all(feature = "hf-hub", not(feature = "local-only")))]
    {
        download_model_files(repo_or_path.as_ref().to_string_lossy().as_ref(), token, subfolder)
    }
    #[cfg(feature = "local-only")]
    {
        Err(anyhow!(
            "remote model downloads are disabled by the `local-only` feature; pass a local model directory instead"
        ))
    }
    #[cfg(all(not(feature = "hf-hub"), not(feature = "local-only")))]
    {
        Err(anyhow!(
            "remote model downloads require the `hf-hub` feature; pass a local model directory instead"
        ))
    }
}

#[cfg(all(feature = "hf-hub", not(feature = "local-only")))]
fn download_model_files(repo_id: &str, token: Option<&str>, subfolder: Option<&str>) -> Result<ModelFiles> {
    let api = match token {
        Some(token) => ApiBuilder::new().with_token(Some(token.to_owned())).build(),
        None => Api::new(),
    }
    .context("hf-hub API init failed")?;
    let repo = api.model(repo_id.to_owned());
    let prefix = subfolder.map(|s| format!("{s}/")).unwrap_or_default();
    resolve_hub_model_files(&repo, &prefix).with_context(|| format!("could not load '{repo_id}' from HuggingFace Hub"))
}

#[cfg(test)]
mod tests {
    use super::{Embeddings, StaticModel, decode_token_mapping};
    use safetensors::tensor::Dtype;
    use std::borrow::Cow;

    #[test]
    fn decode_token_mapping_supports_i32_and_i64() {
        let i32_raw = [1i32, 2, 3]
            .into_iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let i64_raw = [4i64, 5, 6]
            .into_iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();

        assert_eq!(decode_token_mapping(Dtype::I32, &i32_raw).unwrap(), vec![1, 2, 3]);
        assert_eq!(decode_token_mapping(Dtype::I64, &i64_raw).unwrap(), vec![4, 5, 6]);
    }

    #[test]
    fn decode_token_mapping_rejects_unsupported_dtype() {
        let err = decode_token_mapping(Dtype::F32, &[0, 0, 0, 0]).unwrap_err();
        assert!(err.to_string().contains("unsupported mapping dtype"));
    }

    #[test]
    fn int8_models_keep_i8_storage() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
        let int8 = StaticModel::from_pretrained(format!("{dir}/test-model-int8"), None, None, None).unwrap();
        assert!(matches!(&int8.inner.embeddings, Embeddings::I8(Cow::Owned(t)) if t.len() == 29528 * 2));
        let f32_model = StaticModel::from_pretrained(format!("{dir}/test-model-float32"), None, None, None).unwrap();
        assert!(matches!(&f32_model.inner.embeddings, Embeddings::F32(_)));
    }
}
