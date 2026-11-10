# model2vec-rs (Python)

Python bindings for [model2vec-rs](https://github.com/MinishLab/model2vec-rs), the Rust implementation of [Model2Vec](https://github.com/MinishLab/model2vec).

## Installation

```bash
pip install maturin
maturin develop --release   # from this directory
```

## Usage

```python
from model2vec_rs import StaticModel

model = StaticModel.from_pretrained("minishlab/potion-base-8M")

embeddings = model.encode(["Hello world", "Rust is awesome"])  # np.ndarray, shape (2, dim)
embedding = model.encode("Hello world")                        # np.ndarray, shape (dim,)

model.dim  # int
```

`encode` accepts `max_length` (default `512`, `None` for no limit) and `batch_size` (default `1024`), and releases the GIL while encoding.
