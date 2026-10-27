use anyhow::{Context, Result, anyhow};
use half::f16;
#[cfg(all(feature = "hf-hub", not(feature = "local-only")))]
use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};
use ndarray::{Array2, ArrayView2, CowArray, Ix2};
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
    tokenizer: LoadedTokenizer,
    embeddings: CowArray<'static, f32, Ix2>,
    weights: Option<Cow<'static, [f32]>>,
    token_mapping: Option<Cow<'static, [usize]>>,
    normalize: bool,
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
enum Tokenization {
    /// Custom IDs-only WordPiece tokenizer, which needs no `Tokenizer`.
    WordPiece(FastWordPiece),
    /// Custom IDs-only Unigram model on top of the crate's pre-processing.
    Unigram(Tokenizer, FastUnigram),
    /// Pipelines neither custom tokenizer reproduces.
    Crate(Tokenizer),
}

#[derive(Debug)]
struct LoadedTokenizer {
    tokenization: Tokenization,
    median_token_length: usize,
    unk_token_id: Option<usize>,
}

impl LoadedTokenizer {
    fn from_tokenizer(tokenizer: Tokenizer) -> Result<Self> {
        let (median_token_length, unk_token_id) = compute_metadata(&tokenizer)?;
        let tokenization = if let Some(ft) = FastWordPiece::from_tokenizer(&tokenizer) {
            Tokenization::WordPiece(ft)
        } else if let Some(fu) = FastUnigram::from_tokenizer(&tokenizer, unk_token_id) {
            Tokenization::Unigram(tokenizer, fu)
        } else {
            Tokenization::Crate(tokenizer)
        };
        Ok(Self {
            tokenization,
            median_token_length,
            unk_token_id,
        })
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(bytes).map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;
        Self::from_tokenizer(tokenizer)
    }

    /// Like [`Self::from_bytes`], but for WordPiece pipelines reads the
    /// tokenizer from a binary cache in `cache_dir` keyed by the checksum of
    /// `bytes`, writing that cache on a miss. Cache errors are ignored.
    fn from_bytes_cached(bytes: &[u8], cache_dir: Option<&Path>) -> Result<Self> {
        let Some(cache_dir) = cache_dir else {
            return Self::from_bytes(bytes);
        };
        let checksum = checksum(bytes);
        let path = cache_dir.join(format!("wordpiece-{checksum:016x}.bin"));
        if let Some((ft, median_token_length)) = fs::read(&path)
            .ok()
            .and_then(|cached| FastWordPiece::from_cache_bytes(&cached, checksum))
        {
            return Ok(Self {
                unk_token_id: ft.unk_id().map(|id| id as usize),
                tokenization: Tokenization::WordPiece(ft),
                median_token_length,
            });
        }
        let loaded = Self::from_bytes(bytes)?;
        if let Tokenization::WordPiece(ft) = &loaded.tokenization {
            let _ = write_atomic(&path, &ft.to_cache_bytes(checksum, loaded.median_token_length));
        }
        Ok(loaded)
    }
}

/// Stable 64-bit checksum (FNV-1a over little-endian words).
fn checksum(bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = 0xcbf2_9ce4_8422_2325 ^ bytes.len() as u64;
    let (words, tail) = bytes.as_chunks::<8>();
    for w in words {
        h = (h ^ u64::from_le_bytes(*w)).wrapping_mul(PRIME);
    }
    for &b in tail {
        h = (h ^ u64::from(b)).wrapping_mul(PRIME);
    }
    h
}

fn write_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    fs::write(&tmp, data)?;
    fs::rename(&tmp, path).inspect_err(|_| {
        let _ = fs::remove_file(&tmp);
    })
}

/// Directory for the binary tokenizer cache: `MODEL2VEC_CACHE_DIR` if set
/// (empty disables the cache), else `model2vec-rs` in the user cache dir.
fn tokenizer_cache_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MODEL2VEC_CACHE_DIR") {
        return (!dir.is_empty()).then(|| PathBuf::from(dir));
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(PathBuf::from))?;
    Some(base.join("model2vec-rs"))
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
        let tokenizer = LoadedTokenizer::from_bytes(tokenizer_bytes.as_ref())?;
        Self::from_loaded_bytes(tokenizer, model_bytes.as_ref(), config_bytes.as_ref(), normalize)
    }

    fn from_loaded_bytes(
        tokenizer: LoadedTokenizer,
        model_bytes: &[u8],
        config_bytes: &[u8],
        normalize: Option<bool>,
    ) -> Result<Self> {
        // Read normalize default from config.json
        let cfg: Value = serde_json::from_slice(config_bytes).context("failed to parse config.json")?;
        let cfg_norm = cfg.get("normalize").and_then(Value::as_bool).unwrap_or(true);
        let normalize = normalize.unwrap_or(cfg_norm);

        // Load the safetensors
        let safet = SafeTensors::deserialize(model_bytes).context("failed to parse safetensors")?;
        let tensor = safet
            .tensor("embeddings")
            .or_else(|_| safet.tensor("0"))
            .or_else(|_| safet.tensor("embedding.weight"))
            .context("embeddings tensor not found")?;

        let [rows, cols]: [usize; 2] = tensor.shape().try_into().context("embedding tensor is not 2-D")?;
        let raw = tensor.data();
        let floats: Vec<f32> = match tensor.dtype() {
            Dtype::F32 => decode_f32(raw),
            Dtype::F16 => decode_f16(raw),
            Dtype::I8 => raw.iter().map(|&b| f32::from(b as i8)).collect(),
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
            Cow::Owned(floats),
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
        let tokenizer = LoadedTokenizer::from_bytes_cached(&tokenizer_bytes, tokenizer_cache_dir().as_deref())?;
        Self::from_loaded_bytes(tokenizer, &model_bytes, &config_bytes, normalize)
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
            LoadedTokenizer::from_tokenizer(tokenizer)?,
            Cow::Owned(embeddings),
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
            LoadedTokenizer::from_tokenizer(tokenizer)?,
            Cow::Borrowed(embeddings),
            rows,
            cols,
            normalize,
            weights.map(Cow::Borrowed),
            token_mapping.map(Cow::Borrowed),
        )
    }

    fn from_storage(
        tokenizer: LoadedTokenizer,
        embeddings: Cow<'static, [f32]>,
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
        let embeddings = match embeddings {
            Cow::Owned(data) => {
                CowArray::from(Array2::from_shape_vec((rows, cols), data).context("failed to build embeddings array")?)
            }
            Cow::Borrowed(data) => {
                CowArray::from(ArrayView2::from_shape((rows, cols), data).context("failed to build embeddings view")?)
            }
        };
        Ok(Self {
            inner: Arc::new(StaticModelInner {
                tokenizer,
                embeddings,
                weights,
                token_mapping,
                normalize,
            }),
        })
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
            Self::truncate_str(text, max_tok, model.tokenizer.median_token_length)
        });
        match &model.tokenizer.tokenization {
            Tokenization::WordPiece(ft) => TOKEN_IDS.with_borrow_mut(|ids| {
                ids.clear();
                ft.encode_ids_into(text, ids);
                self.pool_ids(ids, max_length)
            }),
            Tokenization::Unigram(tokenizer, fu) => {
                let ids = fu.encode_ids(tokenizer, text).expect("tokenization failed");
                self.pool_ids(&ids, max_length)
            }
            Tokenization::Crate(tokenizer) => {
                let encoding = tokenizer.encode_fast(text, false).expect("tokenization failed");
                self.pool_ids(encoding.get_ids(), max_length)
            }
        }
    }

    /// Mean-pool the first `max_length` token IDs, skipping the unk token.
    fn pool_ids(&self, ids: &[u32], max_length: Option<usize>) -> Vec<f32> {
        let model = &self.inner;
        let ids = &ids[..max_length.map_or(ids.len(), |max_tok| max_tok.min(ids.len()))];
        let (mut sum, cnt) = match model.embeddings.ncols() {
            64 => self.sum_rows_fixed::<64>(ids),
            128 => self.sum_rows_fixed::<128>(ids),
            256 => self.sum_rows_fixed::<256>(ids),
            384 => self.sum_rows_fixed::<384>(ids),
            512 => self.sum_rows_fixed::<512>(ids),
            768 => self.sum_rows_fixed::<768>(ids),
            1024 => self.sum_rows_fixed::<1024>(ids),
            _ => self.sum_rows_dyn(ids),
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

    /// Weighted row sum for a compile-time embedding dimension.
    fn sum_rows_fixed<const D: usize>(&self, ids: &[u32]) -> (Vec<f32>, usize) {
        let mut sum = [0.0_f32; D];
        let cnt = self.for_each_row(ids, |row, scale| {
            let row: &[f32; D] = row.try_into().expect("embedding row has D columns");
            for (s, &v) in sum.iter_mut().zip(row) {
                *s += v * scale;
            }
        });
        (sum.to_vec(), cnt)
    }

    /// Weighted row sum for any embedding dimension.
    fn sum_rows_dyn(&self, ids: &[u32]) -> (Vec<f32>, usize) {
        let mut sum = vec![0.0_f32; self.inner.embeddings.ncols()];
        let cnt = self.for_each_row(ids, |row, scale| {
            for (s, &v) in sum.iter_mut().zip(row) {
                *s += v * scale;
            }
        });
        (sum, cnt)
    }

    /// Call `f` with the embedding row and weight of every non-unk token;
    /// returns the number of rows visited.
    #[inline(always)]
    fn for_each_row(&self, ids: &[u32], mut f: impl FnMut(&[f32], f32)) -> usize {
        let model = &self.inner;
        let table = model.embeddings.as_slice().expect("embeddings are contiguous");
        let dim = model.embeddings.ncols();
        let mut cnt = 0;
        let tokens = ids
            .iter()
            .map(|&id| id as usize)
            .filter(|&tok| Some(tok) != model.tokenizer.unk_token_id);
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
    use super::{LoadedTokenizer, Tokenization, checksum, decode_token_mapping};
    use safetensors::tensor::Dtype;
    use std::fs;

    #[test]
    fn cached_tokenizer_matches_uncached() {
        let fixture = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/test-model-int8/tokenizer.json"
        );
        let bytes = fs::read(fixture).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let cache_file = dir.path().join(format!("wordpiece-{:016x}.bin", checksum(&bytes)));

        let miss = LoadedTokenizer::from_bytes_cached(&bytes, Some(dir.path())).unwrap();
        assert!(cache_file.exists());
        let hit = LoadedTokenizer::from_bytes_cached(&bytes, Some(dir.path())).unwrap();
        let (Tokenization::WordPiece(a), Tokenization::WordPiece(b)) = (&miss.tokenization, &hit.tokenization) else {
            panic!("fixture is a WordPiece pipeline");
        };
        assert_eq!(miss.median_token_length, hit.median_token_length);
        assert_eq!(miss.unk_token_id, hit.unk_token_id);
        for text in ["Hello, world!", "café [MASK] 你好", "supercalifragilistic"] {
            assert_eq!(a.encode_ids(text), b.encode_ids(text));
        }

        // A cache file with another checksum is ignored and overwritten.
        fs::write(&cache_file, a.to_cache_bytes(checksum(&bytes) ^ 1, 1)).unwrap();
        let reloaded = LoadedTokenizer::from_bytes_cached(&bytes, Some(dir.path())).unwrap();
        assert_eq!(reloaded.median_token_length, miss.median_token_length);
        assert_eq!(
            fs::read(&cache_file).unwrap(),
            a.to_cache_bytes(checksum(&bytes), miss.median_token_length)
        );
    }

    #[test]
    fn checksum_changes_with_content() {
        assert_ne!(checksum(b"abcdefgh"), checksum(b"abcdefgi"));
        assert_ne!(checksum(b""), checksum(b"\0"));
        assert_eq!(checksum(b"tokenizer"), checksum(b"tokenizer"));
    }

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
}
