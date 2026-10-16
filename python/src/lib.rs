use std::str::FromStr;

use model2vec_rs::model::StaticModel as RustStaticModel;
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;
use tokenizers::Tokenizer;

#[derive(FromPyObject)]
enum Sentences {
    Single(String),
    Many(Vec<String>),
}

/// Static embedding model for Model2Vec.
#[pyclass(module = "model2vec_rs", name = "StaticModel", frozen)]
struct StaticModel {
    model: RustStaticModel,
    tokenizer: PyOnceLock<Py<PyAny>>,
}

fn read_only<'py>(array: Bound<'py, PyArray2<f32>>) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let kwargs = PyDict::new(array.py());
    kwargs.set_item("write", false)?;
    array.call_method("setflags", (), Some(&kwargs))?;
    Ok(array)
}

#[pymethods]
impl StaticModel {
    /// Load a Model2Vec model from a local folder or the Hugging Face Hub.
    ///
    /// Args:
    ///     path: Hugging Face repo id or path to a local model folder.
    ///     token: Optional Hugging Face token for private models.
    ///     normalize: Override the normalization setting from the model config.
    ///     subfolder: Optional subfolder within the repo or folder.
    #[staticmethod]
    #[pyo3(signature = (path, token=None, normalize=None, subfolder=None))]
    fn from_pretrained(
        py: Python<'_>,
        path: &str,
        token: Option<&str>,
        normalize: Option<bool>,
        subfolder: Option<&str>,
    ) -> PyResult<Self> {
        let model = py.detach(|| RustStaticModel::from_pretrained(path, token, normalize, subfolder))?;
        Ok(Self {
            model,
            tokenizer: PyOnceLock::new(),
        })
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
        let model = RustStaticModel::from_owned(
            tokenizer,
            embeddings.iter().copied().collect(),
            rows,
            cols,
            normalize,
            weights.map(|w| w.as_array().to_vec()),
            token_mapping,
        )?;
        Ok(Self {
            model,
            tokenizer: PyOnceLock::new(),
        })
    }

    /// Encode one or more sentences into embeddings.
    ///
    /// Args:
    ///     sentences: A single string or a sequence of strings.
    ///     max_length: Maximum number of tokens per sentence, or None for no limit.
    ///     batch_size: Number of sentences tokenized at once.
    ///
    /// Returns:
    ///     A float32 array of shape (dim,) for a single string, or (n, dim) for a sequence.
    #[pyo3(signature = (sentences, max_length=Some(512), batch_size=1024))]
    fn encode<'py>(
        &self,
        py: Python<'py>,
        sentences: Sentences,
        max_length: Option<usize>,
        batch_size: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be greater than 0"));
        }
        let dim = self.model.embeddings().ncols();
        match sentences {
            Sentences::Single(sentence) => {
                let embedding = py.detach(|| {
                    self.model
                        .encode_with_args(&[sentence], max_length, batch_size)
                        .pop()
                        .unwrap_or_default()
                });
                Ok(Array1::from_vec(embedding).into_pyarray(py).into_any())
            }
            Sentences::Many(sentences) => {
                let flat = py.detach(|| self.model.encode_with_args(&sentences, max_length, batch_size).concat());
                let rows = flat.len() / dim;
                let array =
                    Array2::from_shape_vec((rows, dim), flat).map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(array.into_pyarray(py).into_any())
            }
        }
    }

    /// The embedding matrix as a read-only float32 array of shape (rows, dim).
    ///
    /// The array shares memory with the model.
    #[getter]
    fn embedding<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let view = slf.get().model.embeddings();
        let array = unsafe { PyArray2::borrow_from_array(&view, slf.clone().into_any()) };
        read_only(array)
    }

    /// Per-token weights as a float32 array, or None if the model has none.
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.model.weights().map(|w| PyArray1::from_slice(py, w))
    }

    /// Mapping from token id to embedding row as an int64 array, or None if the model has none.
    #[getter]
    fn token_mapping<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<i64>>> {
        self.model
            .token_mapping()
            .map(|m| m.iter().map(|&i| i as i64).collect::<Vec<_>>().into_pyarray(py))
    }

    /// The tokenizer as a `tokenizers.Tokenizer`.
    ///
    /// This is a copy of the Rust tokenizer; changing it does not affect `encode`.
    #[getter]
    fn tokenizer<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.tokenizer
            .get_or_try_init(py, || {
                let json = self
                    .model
                    .tokenizer()
                    .to_string(false)
                    .map_err(|e| PyValueError::new_err(format!("failed to serialize tokenizer: {e}")))?;
                let tokenizer = py
                    .import("tokenizers")?
                    .getattr("Tokenizer")?
                    .call_method1("from_str", (json,))?;
                Ok::<_, PyErr>(tokenizer.unbind())
            })
            .map(|t| t.bind(py).clone())
    }

    /// Whether output embeddings are L2-normalized.
    #[getter]
    fn normalize(&self) -> bool {
        self.model.normalize()
    }

    /// The embedding dimension.
    #[getter]
    fn dim(&self) -> usize {
        self.model.embeddings().ncols()
    }

    fn __repr__(&self) -> String {
        let embeddings = self.model.embeddings();
        format!(
            "StaticModel(vocab_size={}, dim={}, normalize={})",
            embeddings.nrows(),
            embeddings.ncols(),
            if self.model.normalize() { "True" } else { "False" }
        )
    }
}

#[pymodule]
#[pyo3(name = "model2vec_rs")]
fn model2vec_rs_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StaticModel>()?;
    Ok(())
}
