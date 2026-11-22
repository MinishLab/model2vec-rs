import gc
import json
from importlib.metadata import version
import multiprocessing
import shutil
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
    return np.asarray(json.loads((FIXTURES / name).read_text(encoding="utf-8")), dtype=np.float32)


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


def test_encode_normalize() -> None:
    texts = ["hello world", "foo", ""]
    normalized = StaticModel.from_pretrained(str(FIXTURES / "test-model-float32"), normalize=True)
    unnormalized = StaticModel.from_pretrained(str(FIXTURES / "test-model-float32"), normalize=False)
    assert normalized.normalize is True
    assert unnormalized.normalize is False
    for model, other, flag in [(normalized, unnormalized, False), (unnormalized, normalized, True)]:
        np.testing.assert_array_equal(model.encode(texts, normalize=None), model.encode(texts))
        np.testing.assert_array_equal(model.encode(texts, normalize=flag), other.encode(texts))
        np.testing.assert_array_equal(model.encode(texts, normalize=flag, batch_size=2), other.encode(texts))
        np.testing.assert_array_equal(model.encode("hello world", normalize=flag), other.encode("hello world"))


LONG_TEXT = " ".join(["hello world"] * 400 + ["foo bar baz"])


def model_with_config_max_length(tmp_path: Path, max_length: int | None) -> Path:
    shutil.copytree(FIXTURES / "test-model-float32", tmp_path / "model")
    config_path = tmp_path / "model" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["max_length"] = max_length
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return tmp_path / "model"


@pytest.mark.parametrize("config_max_length", [None, 7])
def test_max_length_from_config(tmp_path: Path, config_max_length: int | None) -> None:
    path = model_with_config_max_length(tmp_path, config_max_length)
    model = StaticModel.from_pretrained(path)
    assert model.max_length == config_max_length
    texts = [LONG_TEXT, "hello world"]
    np.testing.assert_array_equal(model.encode(texts), model.encode(texts, max_length=config_max_length))
    np.testing.assert_array_equal(model.encode(LONG_TEXT), model.encode(LONG_TEXT, max_length=config_max_length))
    other = 3 if config_max_length is None else None
    assert not np.array_equal(model.encode(LONG_TEXT), model.encode(LONG_TEXT, max_length=other))


def test_max_length_default(model: StaticModel) -> None:
    assert model.max_length == 512
    unlimited = model.encode(LONG_TEXT, max_length=None)
    assert not np.array_equal(model.encode(LONG_TEXT), unlimited)
    np.testing.assert_array_equal(model.encode(LONG_TEXT), model.encode(LONG_TEXT, max_length=512))


@pytest.mark.parametrize("override", [None, 3, 600])
def test_from_pretrained_max_length_override(tmp_path: Path, override: int | None) -> None:
    path = model_with_config_max_length(tmp_path, 7)
    model = StaticModel.from_pretrained(path, max_length=override)
    assert model.max_length == override
    np.testing.assert_array_equal(model.encode(LONG_TEXT), model.encode(LONG_TEXT, max_length=override))


def test_from_arrays_max_length() -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text(encoding="utf-8")
    embeddings = np.random.default_rng(0).random((tokenizers_vocab_size(tokenizer), 4), dtype=np.float32)
    assert StaticModel.from_arrays(tokenizer, embeddings).max_length == 512
    for max_length in [None, 7]:
        model = StaticModel.from_arrays(tokenizer, embeddings, max_length=max_length)
        assert model.max_length == max_length
        np.testing.assert_array_equal(model.encode(LONG_TEXT), model.encode(LONG_TEXT, max_length=max_length))


def test_invalid_config_max_length(tmp_path: Path) -> None:
    path = model_with_config_max_length(tmp_path, -1)
    with pytest.raises(RuntimeError, match="max_length"):
        StaticModel.from_pretrained(path)


def test_repr(model: StaticModel) -> None:
    assert repr(model).startswith("StaticModel(dim=")


def test_from_pretrained_path(model: StaticModel) -> None:
    from_path = StaticModel.from_pretrained(FIXTURES / "test-model-float32")
    np.testing.assert_array_equal(from_path.encode(["hello world"]), model.encode(["hello world"]))


def test_version() -> None:
    assert model2vec_rs.__version__ == version("model2vec-rs")


def test_from_arrays() -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text(encoding="utf-8")
    vocab_size = tokenizers_vocab_size(tokenizer)
    embeddings = np.random.default_rng(0).random((vocab_size, 4), dtype=np.float32)
    assert StaticModel.from_arrays(tokenizer, embeddings).encode(["hello world"]).shape == (1, 4)
    mapping = np.arange(vocab_size, dtype=np.int64) % 10
    model = StaticModel.from_arrays(tokenizer, embeddings[:10], token_mapping=mapping)
    assert model.encode(["hello world"]).shape == (1, 4)


@pytest.mark.parametrize("vocab_quantized", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_from_arrays_int8_matches_float32(vocab_quantized: bool, normalize: bool) -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text(encoding="utf-8")
    vocab_size = tokenizers_vocab_size(tokenizer)
    rng = np.random.default_rng(0)
    rows = 10 if vocab_quantized else vocab_size
    embeddings = rng.integers(-128, 128, size=(rows, 4), dtype=np.int8)
    weights = rng.random(vocab_size, dtype=np.float32)
    mapping = np.arange(vocab_size, dtype=np.int64) % rows if vocab_quantized else None
    texts = ["hello world", "foo bar baz", ""]
    int8 = StaticModel.from_arrays(tokenizer, embeddings, normalize=normalize, weights=weights, token_mapping=mapping)
    float32 = StaticModel.from_arrays(
        tokenizer, embeddings.astype(np.float32), normalize=normalize, weights=weights, token_mapping=mapping
    )
    np.testing.assert_allclose(int8.encode(texts), float32.encode(texts), rtol=1e-6)


def random_arrays(dtype: type, vocab_size: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if dtype is np.int8:
        embeddings = rng.integers(-128, 128, size=(vocab_size, 8), dtype=np.int8)
    else:
        embeddings = rng.standard_normal((vocab_size, 8), dtype=np.float32)
    return embeddings, rng.random(vocab_size, dtype=np.float32)


def read_only(array: np.ndarray, order: str = "C") -> np.ndarray:
    data = array.tobytes(order=order)
    return np.frombuffer(data, dtype=array.dtype).reshape(array.shape, order=order)


def mapped(array: np.ndarray, path: Path) -> np.ndarray:
    np.save(path, array)
    return np.load(path, mmap_mode="r")


ARRAY_TEXTS = ["hello world", "foo bar baz", "the quick brown fox jumps over the lazy dog", ""]


@pytest.mark.parametrize("dtype", [np.float32, np.int8])
@pytest.mark.parametrize("source", ["bytes", "mmap"])
def test_from_arrays_shares_read_only_arrays(tmp_path: Path, dtype: type, source: str) -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text(encoding="utf-8")
    vocab_size = tokenizers_vocab_size(tokenizer)
    embeddings, weights = random_arrays(dtype, vocab_size)
    want = StaticModel.from_arrays(tokenizer, embeddings, weights=weights).encode(ARRAY_TEXTS)
    if source == "bytes":
        shared_embeddings, shared_weights = read_only(embeddings), read_only(weights)
    else:
        shared_embeddings = mapped(embeddings, tmp_path / "embeddings.npy")
        shared_weights = mapped(weights, tmp_path / "weights.npy")
    roots = [shared_embeddings.base, shared_weights.base]
    with_array = [sys.getrefcount(root) for root in roots]

    model = StaticModel.from_arrays(tokenizer, shared_embeddings, weights=shared_weights)
    del shared_embeddings, shared_weights
    gc.collect()

    assert [sys.getrefcount(root) for root in roots] == with_array
    np.testing.assert_array_equal(model.encode(ARRAY_TEXTS), want)
    del model
    gc.collect()
    without_model = [sys.getrefcount(root) for root in roots]
    assert all(after < before for after, before in zip(without_model, with_array))


@pytest.mark.parametrize("dtype", [np.float32, np.int8])
@pytest.mark.parametrize("source", ["bytes", "mmap"])
def test_from_arrays_keeps_shared_arrays_alive(tmp_path: Path, dtype: type, source: str) -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text(encoding="utf-8")
    vocab_size = tokenizers_vocab_size(tokenizer)
    embeddings, weights = random_arrays(dtype, vocab_size)
    want = StaticModel.from_arrays(tokenizer, embeddings, weights=weights).encode(ARRAY_TEXTS)
    if source == "bytes":
        shared_embeddings, shared_weights = read_only(embeddings), read_only(weights)
    else:
        shared_embeddings = mapped(embeddings, tmp_path / "embeddings.npy")
        shared_weights = mapped(weights, tmp_path / "weights.npy")

    model = StaticModel.from_arrays(tokenizer, shared_embeddings, weights=shared_weights)
    del shared_embeddings, shared_weights, embeddings, weights
    gc.collect()
    garbage = [np.full((vocab_size, 8), 7, dtype=dtype) for _ in range(8)]

    np.testing.assert_array_equal(model.encode(ARRAY_TEXTS), want)
    del garbage


@pytest.mark.parametrize("dtype", [np.float32, np.int8])
@pytest.mark.parametrize("source", ["array", "unflagged", "bytearray", "readonly_view", "copy_on_write"])
def test_from_arrays_copies_writable_arrays(tmp_path: Path, dtype: type, source: str) -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text(encoding="utf-8")
    vocab_size = tokenizers_vocab_size(tokenizer)
    embeddings, weights = random_arrays(dtype, vocab_size)
    want = StaticModel.from_arrays(tokenizer, embeddings.copy(), weights=weights.copy()).encode(ARRAY_TEXTS)
    if source == "array":
        writable = [embeddings, weights]
    elif source == "unflagged":
        embeddings.flags.writeable = False
        weights.flags.writeable = False
        writable = [embeddings, weights]
    elif source == "bytearray":
        buffers = [bytearray(embeddings.tobytes()), bytearray(weights.tobytes())]
        embeddings = np.frombuffer(buffers[0], dtype=dtype).reshape(embeddings.shape)
        weights = np.frombuffer(buffers[1], dtype=np.float32)
        writable = [embeddings, weights]
    elif source == "readonly_view":
        buffers = [bytearray(embeddings.tobytes()), bytearray(weights.tobytes())]
        views = [memoryview(buffer).toreadonly() for buffer in buffers]
        embeddings = np.frombuffer(views[0], dtype=dtype).reshape(embeddings.shape)
        weights = np.frombuffer(views[1], dtype=np.float32)
        assert not embeddings.flags.writeable
        writable = [np.frombuffer(buffer, dtype=array.dtype) for buffer, array in zip(buffers, [embeddings, weights])]
    else:
        np.save(tmp_path / "embeddings.npy", embeddings)
        np.save(tmp_path / "weights.npy", weights)
        embeddings = np.load(tmp_path / "embeddings.npy", mmap_mode="c")
        weights = np.load(tmp_path / "weights.npy", mmap_mode="c")
        writable = [embeddings, weights]

    model = StaticModel.from_arrays(tokenizer, embeddings, weights=weights)
    for array in writable:
        array.flags.writeable = True
        array[...] = 0

    np.testing.assert_array_equal(model.encode(ARRAY_TEXTS), want)


@pytest.mark.parametrize("dtype", [np.float32, np.int8])
@pytest.mark.parametrize("layout", ["fortran", "strided", "reversed"])
@pytest.mark.parametrize("readonly", [False, True])
def test_from_arrays_copies_non_contiguous_arrays(dtype: type, layout: str, readonly: bool) -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text(encoding="utf-8")
    vocab_size = tokenizers_vocab_size(tokenizer)
    embeddings, weights = random_arrays(dtype, vocab_size)
    wrap = read_only if readonly else np.ascontiguousarray
    if layout == "fortran":
        odd_embeddings = read_only(embeddings, order="F") if readonly else np.asfortranarray(embeddings)
        odd_weights = wrap(np.repeat(weights, 2))[::2]
    elif layout == "strided":
        odd_embeddings = wrap(np.repeat(embeddings, 2, axis=1))[:, ::2]
        odd_weights = wrap(np.repeat(weights, 2))[::2]
    else:
        odd_embeddings = wrap(embeddings[::-1, ::-1])[::-1, ::-1]
        odd_weights = wrap(weights[::-1])[::-1]
    assert not odd_embeddings.flags.c_contiguous
    assert not odd_weights.flags.c_contiguous
    assert odd_embeddings.flags.writeable != readonly
    np.testing.assert_array_equal(odd_embeddings, embeddings)
    np.testing.assert_array_equal(odd_weights, weights)

    want = StaticModel.from_arrays(tokenizer, embeddings, weights=weights).encode(ARRAY_TEXTS)
    got = StaticModel.from_arrays(tokenizer, odd_embeddings, weights=odd_weights).encode(ARRAY_TEXTS)

    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("dtype", [np.float32, np.int8])
def test_from_arrays_exposes_read_only_arrays(dtype: type) -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text(encoding="utf-8")
    vocab_size = tokenizers_vocab_size(tokenizer)
    embeddings, weights = random_arrays(dtype, vocab_size)
    model = StaticModel.from_arrays(tokenizer, embeddings, weights=weights)

    for got, original in [(model.embeddings, embeddings), (model.weights, weights)]:
        np.testing.assert_array_equal(got, original)
        assert got.dtype == original.dtype
        assert got.flags.c_contiguous
        assert not np.shares_memory(got, original)
        with pytest.raises(ValueError, match="read-only"):
            got[0] = 0
        with pytest.raises(ValueError):
            got.flags.writeable = True


@pytest.mark.parametrize("dtype", [np.float32, np.int8])
def test_from_arrays_shares_arrays_of_another_model(dtype: type) -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text(encoding="utf-8")
    vocab_size = tokenizers_vocab_size(tokenizer)
    embeddings, weights = random_arrays(dtype, vocab_size)
    first = StaticModel.from_arrays(tokenizer, embeddings, weights=weights)
    want = first.encode(ARRAY_TEXTS)

    second = StaticModel.from_arrays(tokenizer, first.embeddings, weights=first.weights)
    assert np.shares_memory(second.embeddings, first.embeddings)
    assert np.shares_memory(second.weights, first.weights)
    del first, embeddings, weights
    gc.collect()
    garbage = [np.full((vocab_size, 8), 7, dtype=dtype) for _ in range(8)]

    np.testing.assert_array_equal(second.encode(ARRAY_TEXTS), want)
    del garbage


@pytest.mark.parametrize("dtype", [np.float32, np.int8])
@pytest.mark.parametrize("source", ["bytes", "mmap"])
def test_from_arrays_exposes_shared_arrays(tmp_path: Path, dtype: type, source: str) -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text(encoding="utf-8")
    embeddings, weights = random_arrays(dtype, tokenizers_vocab_size(tokenizer))
    if source == "bytes":
        shared_embeddings, shared_weights = read_only(embeddings), read_only(weights)
    else:
        shared_embeddings = mapped(embeddings, tmp_path / "embeddings.npy")
        shared_weights = mapped(weights, tmp_path / "weights.npy")
    model = StaticModel.from_arrays(tokenizer, shared_embeddings, weights=shared_weights)

    assert np.shares_memory(model.embeddings, shared_embeddings)
    assert np.shares_memory(model.weights, shared_weights)
    assert not model.embeddings.flags.writeable
    assert not model.weights.flags.writeable


def test_from_arrays_arrays_outlive_model() -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text(encoding="utf-8")
    embeddings, weights = random_arrays(np.float32, tokenizers_vocab_size(tokenizer))
    model = StaticModel.from_arrays(tokenizer, embeddings, weights=weights)
    got_embeddings, got_weights = model.embeddings, model.weights
    del model
    gc.collect()

    np.testing.assert_array_equal(got_embeddings, embeddings)
    np.testing.assert_array_equal(got_weights, weights)


def test_from_arrays_without_weights_has_no_weights() -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text(encoding="utf-8")
    embeddings, _ = random_arrays(np.float32, tokenizers_vocab_size(tokenizer))
    assert StaticModel.from_arrays(tokenizer, embeddings).weights is None


def test_from_pretrained_has_no_arrays(model: StaticModel) -> None:
    assert model.embeddings is None
    assert model.weights is None


def test_from_arrays_rejects_other_dtypes() -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text(encoding="utf-8")
    vocab_size = tokenizers_vocab_size(tokenizer)
    with pytest.raises(TypeError):
        StaticModel.from_arrays(tokenizer, np.zeros((vocab_size, 4), dtype=np.float64))


def test_from_arrays_too_few_rows() -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="rows"):
        StaticModel.from_arrays(tokenizer, np.zeros((10, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="token_mapping"):
        StaticModel.from_arrays(
            tokenizer, np.zeros((10, 4), dtype=np.float32), token_mapping=np.zeros(5, dtype=np.int64)
        )


def test_from_arrays_too_few_weights() -> None:
    tokenizer = (FIXTURES / "test-model-float32" / "tokenizer.json").read_text(encoding="utf-8")
    vocab_size = tokenizers_vocab_size(tokenizer)
    embeddings = np.zeros((vocab_size, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="weights"):
        StaticModel.from_arrays(tokenizer, embeddings, weights=np.ones(vocab_size - 1, dtype=np.float32))
    with pytest.raises(ValueError, match="weights"):
        StaticModel.from_arrays(
            tokenizer,
            embeddings[:10],
            weights=np.ones(3, dtype=np.float32),
            token_mapping=np.zeros(vocab_size, dtype=np.int64),
        )
    StaticModel.from_arrays(tokenizer, embeddings, weights=np.ones(vocab_size, dtype=np.float32))


def test_missing_model() -> None:
    with pytest.raises(RuntimeError):
        StaticModel.from_pretrained("/nonexistent/path/to/model")
