import json
from importlib.metadata import version
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pytest

import model2vec_rs
from model2vec_rs import StaticModel

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


@pytest.fixture(scope="module")
def model() -> StaticModel:
    return StaticModel.from_pretrained(str(FIXTURES / "test-model-float32"))


@pytest.fixture(scope="module")
def quantized_model() -> StaticModel:
    return StaticModel.from_pretrained(str(FIXTURES / "test-model-vocab-quantized"))


def tokenizers_vocab_size(tokenizer: str) -> int:
    parsed = json.loads(tokenizer)
    ids = list(parsed["model"]["vocab"].values()) + [token["id"] for token in parsed["added_tokens"]]
    return max(ids) + 1


def load_fixture(name: str) -> np.ndarray:
    return np.asarray(json.loads((FIXTURES / name).read_text()), dtype=np.float32)


def test_encode_matches_python(model: StaticModel) -> None:
    np.testing.assert_allclose(model.encode(["hello world"]), load_fixture("embeddings_short.json"), rtol=1e-5)
    long_text = " ".join(["hello"] * 1000)
    np.testing.assert_allclose(model.encode([long_text]), load_fixture("embeddings_long.json"), rtol=1e-5)


def test_encode_matches_python_vocab_quantized(quantized_model: StaticModel) -> None:
    long_text = " ".join(["hello"] * 1000)
    np.testing.assert_allclose(
        quantized_model.encode([long_text]), load_fixture("embeddings_vocab_quantized.json"), rtol=1e-5
    )


def test_encode_shapes(model: StaticModel) -> None:
    batch = model.encode(["hello world", "foo bar", "baz"])
    assert batch.shape == (3, model.dim)
    assert batch.dtype == np.float32

    single = model.encode("hello world")
    assert single.shape == (model.dim,)
    np.testing.assert_array_equal(single, batch[0])

    assert model.encode([]).shape == (0, model.dim)
    assert model.encode(("hello world",)).shape == (1, model.dim)


def test_encode_args(model: StaticModel) -> None:
    texts = ["hello world"] * 5
    np.testing.assert_array_equal(model.encode(texts, batch_size=2), model.encode(texts))
    np.testing.assert_array_equal(model.encode(texts, batch_size=None), model.encode(texts))
    assert model.encode(texts, max_length=None).shape == (5, model.dim)
    with pytest.raises(ValueError):
        model.encode(texts, batch_size=0)
    with pytest.raises(TypeError):
        model.encode([1, 2])


def encode_batch(model: StaticModel) -> None:
    model.encode(["hello world"] * 1000)


@pytest.mark.skipif(sys.platform == "win32", reason="fork is not available on Windows")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_encode_after_fork(model: StaticModel) -> None:
    encode_batch(model)
    process = multiprocessing.get_context("fork").Process(target=encode_batch, args=(model,))
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("encode hung in a forked child")
    assert process.exitcode == 0
    encode_batch(model)


def test_normalize_override() -> None:
    model = StaticModel.from_pretrained(str(FIXTURES / "test-model-float32"), normalize=True)
    np.testing.assert_allclose(np.linalg.norm(model.encode(["hello world", "foo"]), axis=1), 1.0, atol=1e-5)


def test_repr(model: StaticModel) -> None:
    assert repr(model).startswith("StaticModel(dim=")


def test_from_pretrained_path(model: StaticModel) -> None:
    from_path = StaticModel.from_pretrained(FIXTURES / "test-model-float32")
    np.testing.assert_array_equal(from_path.encode(["hello world"]), model.encode(["hello world"]))


def test_version() -> None:
    assert model2vec_rs.__version__ == version("model2vec-rs")


def test_from_arrays() -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text()
    vocab_size = tokenizers_vocab_size(tokenizer)
    embeddings = np.random.default_rng(0).random((vocab_size, 4), dtype=np.float32)
    assert StaticModel.from_arrays(tokenizer, embeddings).encode(["hello world"]).shape == (1, 4)
    mapping = np.arange(vocab_size, dtype=np.int64) % 10
    model = StaticModel.from_arrays(tokenizer, embeddings[:10], token_mapping=mapping)
    assert model.encode(["hello world"]).shape == (1, 4)


def test_from_arrays_too_few_rows() -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text()
    with pytest.raises(ValueError, match="rows"):
        StaticModel.from_arrays(tokenizer, np.zeros((10, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="token_mapping"):
        StaticModel.from_arrays(
            tokenizer, np.zeros((10, 4), dtype=np.float32), token_mapping=np.zeros(5, dtype=np.int64)
        )


def test_missing_model() -> None:
    with pytest.raises(RuntimeError):
        StaticModel.from_pretrained("/nonexistent/path/to/model")
