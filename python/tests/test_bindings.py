import json
from pathlib import Path

import numpy as np
import pytest
import tokenizers

from model2vec_rs import StaticModel

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


@pytest.fixture(scope="module")
def model() -> StaticModel:
    return StaticModel.from_pretrained(str(FIXTURES / "test-model-float32"))


@pytest.fixture(scope="module")
def quantized_model() -> StaticModel:
    return StaticModel.from_pretrained(str(FIXTURES / "test-model-vocab-quantized"))


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
    assert model.encode(texts, max_length=None).shape == (5, model.dim)
    with pytest.raises(ValueError):
        model.encode(texts, batch_size=0)
    with pytest.raises(TypeError):
        model.encode([1, 2])


def test_embedding(model: StaticModel) -> None:
    embedding = model.embedding
    assert embedding.dtype == np.float32
    assert embedding.shape == (model.tokenizer.get_vocab_size(), model.dim)
    assert not embedding.flags.writeable
    with pytest.raises(ValueError):
        embedding[0, 0] = 1.0
    assert np.shares_memory(embedding, model.embedding)


def test_embedding_outlives_model() -> None:
    embedding = StaticModel.from_pretrained(str(FIXTURES / "test-model-float32")).embedding
    expected = StaticModel.from_pretrained(str(FIXTURES / "test-model-float32")).embedding
    np.testing.assert_array_equal(embedding, expected)


def test_embedding_matches_encode(model: StaticModel) -> None:
    token_id = model.tokenizer.token_to_id("hello")
    expected = model.embedding[token_id]
    if model.normalize:
        expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(model.encode("hello"), expected, rtol=1e-5)


def test_tokenizer(model: StaticModel) -> None:
    tokenizer = model.tokenizer
    assert isinstance(tokenizer, tokenizers.Tokenizer)
    assert model.tokenizer is tokenizer
    assert tokenizer.encode("hello world", add_special_tokens=False).ids


def test_quantized_attributes(model: StaticModel, quantized_model: StaticModel) -> None:
    assert model.token_mapping is None
    mapping = quantized_model.token_mapping
    assert mapping is not None
    assert mapping.dtype == np.int64
    assert mapping.max() < quantized_model.embedding.shape[0]


def test_normalize_override() -> None:
    model = StaticModel.from_pretrained(str(FIXTURES / "test-model-float32"), normalize=True)
    assert model.normalize
    np.testing.assert_allclose(np.linalg.norm(model.encode(["hello world", "foo"]), axis=1), 1.0, atol=1e-5)


def test_repr(model: StaticModel) -> None:
    assert repr(model).startswith("StaticModel(vocab_size=")


def test_missing_model() -> None:
    with pytest.raises(RuntimeError):
        StaticModel.from_pretrained("/nonexistent/path/to/model")
