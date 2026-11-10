use std::path::PathBuf;
use std::str::FromStr;

use model2vec_rs::model::StaticModel as RustStaticModel;
use ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokenizers::Tokenizer;
use tokenizers::utils::parallelism;

#[derive(FromPyObject)]
enum Sentences {
    Single(String),
    Many(Vec<String>),
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
    #[staticmethod]
    #[pyo3(signature = (path, token=None, normalize=None, subfolder=None))]
    fn from_pretrained(
        py: Python<'_>,
        path: PathBuf,
        token: Option<&str>,
        normalize: Option<bool>,
        subfolder: Option<&str>,
    ) -> PyResult<Self> {
        let model = py.detach(|| RustStaticModel::from_pretrained(path, token, normalize, subfolder))?;
        Ok(Self { model })
    }

    /// Build a model from in-memory data.
    ///
    /// Args:
    ///     tokenizer: The tokenizer serialized as a JSON string.
    ///     embeddings: A float32 array of shape (rows, dim).
    ///     normalize: Whether to L2-normalize output embeddings.
    ///     weights: Optional float32 per-token weights, indexed by token id.
    ///     token_mapping: Optional int64 mapping from token id to embedding row.
    #[staticmethod]
    #[pyo3(signature = (tokenizer, embeddings, normalize=false, weights=None, token_mapping=None))]
    fn from_arrays(
        tokenizer: &str,
        embeddings: PyReadonlyArray2<'_, f32>,
        normalize: bool,
        weights: Option<PyReadonlyArray1<'_, f32>>,
        token_mapping: Option<PyReadonlyArray1<'_, i64>>,
    ) -> PyResult<Self> {
        let tokenizer = Tokenizer::from_str(tokenizer)
            .map_err(|e| PyValueError::new_err(format!("failed to load tokenizer: {e}")))?;
        let embeddings = embeddings.as_array();
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
        let model = RustStaticModel::from_owned(
            tokenizer,
            embeddings.iter().copied().collect(),
            rows,
            cols,
            normalize,
            weights.map(|w| w.as_array().to_vec()),
            token_mapping,
        )?;
        Ok(Self { model })
    }

    /// Encode one or more sentences into embeddings.
    ///
    /// Args:
    ///     sentences: A single string or a sequence of strings.
    ///     max_length: Maximum number of tokens per sentence, or None for no limit.
    ///     batch_size: Number of sentences encoded per parallel pass, or None to encode all at once.
    ///
    /// Returns:
    ///     A float32 array of shape (dim,) for a single string, or (n, dim) for a sequence.
    #[pyo3(signature = (sentences, max_length=Some(512), batch_size=None))]
    fn encode<'py>(
        &self,
        py: Python<'py>,
        sentences: Sentences,
        max_length: Option<usize>,
        batch_size: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if batch_size == Some(0) {
            return Err(PyValueError::new_err("batch_size must be greater than 0"));
        }
        let dim = self.model.dim();
        match sentences {
            Sentences::Single(sentence) => {
                let embedding = py.detach(|| {
                    self.model
                        .encode_flat_with_args(std::slice::from_ref(&sentence), max_length)
                });
                Ok(embedding.into_pyarray(py).into_any())
            }
            Sentences::Many(sentences) => {
                let flat = py.detach(|| match batch_size {
                    None => self.model.encode_flat_with_args(&sentences, max_length),
                    Some(batch_size) => {
                        let mut flat = Vec::with_capacity(sentences.len() * dim);
                        for batch in sentences.chunks(batch_size) {
                            flat.extend(self.model.encode_flat_with_args(batch, max_length));
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
