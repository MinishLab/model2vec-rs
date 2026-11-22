use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use model2vec_rs::model::{DEFAULT_MAX_LENGTH, SharedSlice, StaticModel as RustStaticModel};
use ndarray::{Array2, Dimension};
use numpy::{Element, IntoPyArray, PyReadonlyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokenizers::Tokenizer;
use tokenizers::utils::parallelism;

/// The data of a C-contiguous NumPy array, which it keeps alive. The array must not be written to while
/// a model uses it.
struct NumpySlice<T> {
    _array: Py<PyAny>,
    data: *const T,
    len: usize,
}

unsafe impl<T: Sync> Send for NumpySlice<T> {}
unsafe impl<T: Sync> Sync for NumpySlice<T> {}

impl<T> AsRef<[T]> for NumpySlice<T> {
    fn as_ref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }
}

/// Share the data of a C-contiguous array without copying it, or copy any other array.
fn shared_slice<T: Element + Copy + Sync + Send + 'static, D: Dimension>(
    array: &PyReadonlyArray<'_, T, D>,
) -> SharedSlice<T> {
    match array.as_slice() {
        Ok(data) if array.is_c_contiguous() => Arc::new(NumpySlice {
            _array: array.as_any().clone().unbind(),
            data: data.as_ptr(),
            len: data.len(),
        }),
        _ => Arc::new(array.as_array().iter().copied().collect::<Vec<T>>()),
    }
}

#[derive(FromPyObject)]
enum Sentences {
    Single(String),
    Many(Vec<String>),
}

#[derive(FromPyObject)]
enum EmbeddingArray<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    I8(PyReadonlyArray2<'py, i8>),
}

impl EmbeddingArray<'_> {
    fn dim(&self) -> (usize, usize) {
        match self {
            Self::F32(array) => array.as_array().dim(),
            Self::I8(array) => array.as_array().dim(),
        }
    }
}

enum MaxLength {
    Unset,
    Set(Option<usize>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for MaxLength {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self::Set(obj.extract()?))
    }
}

impl MaxLength {
    fn or(self, default: Option<usize>) -> Option<usize> {
        match self {
            Self::Unset => default,
            Self::Set(max_length) => max_length,
        }
    }
}

/// Static embedding model for Model2Vec.
#[pyclass(module = "model2vec_rs", name = "StaticModel", frozen)]
struct StaticModel {
    model: RustStaticModel,
}

#[pymethods]
impl StaticModel {
    /// Load a Model2Vec model from a local folder or the Hugging Face Hub.
    ///
    /// Args:
    ///     path: Hugging Face repo id or path to a local model folder, as a string or path-like object.
    ///     token: Optional Hugging Face token for private models.
    ///     normalize: Override the normalization setting from the model config.
    ///     subfolder: Optional subfolder within the repo or folder.
    ///     max_length: Override the default maximum number of tokens per sentence used by `encode`,
    ///         or None for no limit. If not passed, the value from the model config is used (512 if unset).
    #[staticmethod]
    #[pyo3(signature = (path, token=None, normalize=None, subfolder=None, max_length=MaxLength::Unset))]
    fn from_pretrained(
        py: Python<'_>,
        path: PathBuf,
        token: Option<&str>,
        normalize: Option<bool>,
        subfolder: Option<&str>,
        max_length: MaxLength,
    ) -> PyResult<Self> {
        let model = py.detach(|| RustStaticModel::from_pretrained(path, token, normalize, subfolder))?;
        let max_length = max_length.or(model.max_length());
        Ok(Self {
            model: model.with_max_length(max_length),
        })
    }

    /// Build a model from in-memory data.
    ///
    /// C-contiguous `embeddings` and `weights` are shared with the model instead of copied, and must not be
    /// modified while the model exists.
    ///
    /// Args:
    ///     tokenizer: The tokenizer serialized as a JSON string.
    ///     embeddings: A float32 or int8 array of shape (rows, dim).
    ///     normalize: Whether to L2-normalize output embeddings.
    ///     weights: Optional float32 per-token weights, indexed by token id.
    ///     token_mapping: Optional int64 mapping from token id to embedding row.
    ///     max_length: The default maximum number of tokens per sentence used by `encode`, or None for no limit.
    #[staticmethod]
    #[pyo3(signature = (tokenizer, embeddings, normalize=false, weights=None, token_mapping=None, max_length=Some(DEFAULT_MAX_LENGTH)))]
    fn from_arrays(
        tokenizer: &str,
        embeddings: EmbeddingArray<'_>,
        normalize: bool,
        weights: Option<PyReadonlyArray1<'_, f32>>,
        token_mapping: Option<PyReadonlyArray1<'_, i64>>,
        max_length: Option<usize>,
    ) -> PyResult<Self> {
        let tokenizer = Tokenizer::from_str(tokenizer)
            .map_err(|e| PyValueError::new_err(format!("failed to load tokenizer: {e}")))?;
        let (rows, cols) = embeddings.dim();
        let vocab_size = tokenizer
            .get_vocab(true)
            .values()
            .max()
            .map_or(0, |&id| id as usize + 1);
        let token_mapping = token_mapping
            .map(|mapping| {
                mapping
                    .as_array()
                    .iter()
                    .map(|&row| {
                        usize::try_from(row)
                            .ok()
                            .filter(|&row| row < rows)
                            .ok_or_else(|| PyValueError::new_err(format!("token_mapping entry {row} is out of range")))
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        match &token_mapping {
            Some(mapping) if mapping.len() < vocab_size => {
                return Err(PyValueError::new_err(format!(
                    "token_mapping has {} entries, but the tokenizer has {vocab_size} tokens",
                    mapping.len()
                )));
            }
            None if rows < vocab_size => {
                return Err(PyValueError::new_err(format!(
                    "embeddings has {rows} rows, but the tokenizer has {vocab_size} tokens"
                )));
            }
            _ => {}
        }
        if let Some(weights) = &weights
            && weights.as_array().len() < vocab_size
        {
            return Err(PyValueError::new_err(format!(
                "weights has {} entries, but the tokenizer has {vocab_size} tokens",
                weights.as_array().len()
            )));
        }
        let weights = weights.map(|w| shared_slice(&w));
        let token_mapping = token_mapping.map(|m| Arc::new(m) as SharedSlice<usize>);
        let model = match embeddings {
            EmbeddingArray::F32(array) => RustStaticModel::from_shared(
                tokenizer,
                shared_slice(&array),
                rows,
                cols,
                normalize,
                weights,
                token_mapping,
            )?,
            EmbeddingArray::I8(array) => RustStaticModel::from_shared_i8(
                tokenizer,
                shared_slice(&array),
                rows,
                cols,
                normalize,
                weights,
                token_mapping,
            )?,
        };
        Ok(Self {
            model: model.with_max_length(max_length),
        })
    }

    /// Encode one or more sentences into embeddings.
    ///
    /// Args:
    ///     sentences: A single string or a sequence of strings.
    ///     max_length: Maximum number of tokens per sentence, or None for no limit. If not passed, the
    ///         model's `max_length` is used.
    ///     batch_size: Number of sentences encoded per parallel pass, or None to encode all at once.
    ///     normalize: Whether to L2-normalize the embeddings, or None to use the model's setting.
    ///
    /// Returns:
    ///     A float32 array of shape (dim,) for a single string, or (n, dim) for a sequence.
    #[pyo3(signature = (sentences, max_length=MaxLength::Unset, batch_size=None, normalize=None))]
    fn encode<'py>(
        &self,
        py: Python<'py>,
        sentences: Sentences,
        max_length: MaxLength,
        batch_size: Option<usize>,
        normalize: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let max_length = max_length.or(self.model.max_length());
        if batch_size == Some(0) {
            return Err(PyValueError::new_err("batch_size must be greater than 0"));
        }
        let dim = self.model.dim();
        match sentences {
            Sentences::Single(sentence) => {
                let embedding = py.detach(|| {
                    self.model
                        .encode_flat_with_args(std::slice::from_ref(&sentence), max_length, normalize)
                });
                Ok(embedding.into_pyarray(py).into_any())
            }
            Sentences::Many(sentences) => {
                let flat = py.detach(|| match batch_size {
                    None => self.model.encode_flat_with_args(&sentences, max_length, normalize),
                    Some(batch_size) => {
                        let mut flat = Vec::with_capacity(sentences.len() * dim);
                        for batch in sentences.chunks(batch_size) {
                            flat.extend(self.model.encode_flat_with_args(batch, max_length, normalize));
                        }
                        flat
                    }
                });
                let array = Array2::from_shape_vec((sentences.len(), dim), flat)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(array.into_pyarray(py).into_any())
            }
        }
    }

    /// The embedding dimension.
    #[getter]
    fn dim(&self) -> usize {
        self.model.dim()
    }

    /// Whether the model L2-normalizes its embeddings by default.
    #[getter]
    fn normalize(&self) -> bool {
        self.model.normalize()
    }

    /// The default maximum number of tokens per sentence used by `encode`, or None for no limit.
    #[getter]
    fn max_length(&self) -> Option<usize> {
        self.model.max_length()
    }

    fn __repr__(&self) -> String {
        format!("StaticModel(dim={})", self.model.dim())
    }
}

/// Disable parallel encoding in a forked child, whose copy of the thread pool has no threads.
#[pyfunction]
fn _disable_parallelism_after_fork() {
    if parallelism::has_parallelism_been_used() {
        parallelism::set_parallelism(false);
    }
}

#[pymodule]
#[pyo3(name = "model2vec_rs")]
fn model2vec_rs_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StaticModel>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    let os = m.py().import("os")?;
    if os.hasattr("register_at_fork")? {
        let kwargs = PyDict::new(m.py());
        kwargs.set_item("after_in_child", wrap_pyfunction!(_disable_parallelism_after_fork, m)?)?;
        os.call_method("register_at_fork", (), Some(&kwargs))?;
    }
    Ok(())
}
