import os
from collections.abc import Sequence
from typing import overload

import numpy as np
import numpy.typing as npt

__version__: str

class StaticModel:
    @staticmethod
    def from_pretrained(
        path: str | os.PathLike[str],
        token: str | None = None,
        normalize: bool | None = None,
        subfolder: str | None = None,
        max_length: int | None = ...,
    ) -> StaticModel: ...
    @staticmethod
    def from_arrays(
        tokenizer: str,
        embeddings: npt.NDArray[np.float32] | npt.NDArray[np.int8],
        normalize: bool = False,
        weights: npt.NDArray[np.float32] | None = None,
        token_mapping: npt.NDArray[np.int64] | None = None,
        max_length: int | None = 512,
    ) -> StaticModel:
        """Build a model from in-memory data.

        `embeddings` and `weights` are copied, unless they are C-contiguous arrays over memory that cannot be
        written to, such as `np.frombuffer(bytes)` or `np.load(..., mmap_mode="r")`: those are shared with the
        model instead. `token_mapping` is always copied.
        """
    @overload
    def encode(
        self,
        sentences: str,
        max_length: int | None = ...,
        batch_size: int | None = None,
        normalize: bool | None = None,
    ) -> npt.NDArray[np.float32]: ...
    @overload
    def encode(
        self,
        sentences: Sequence[str],
        max_length: int | None = ...,
        batch_size: int | None = None,
        normalize: bool | None = None,
    ) -> npt.NDArray[np.float32]: ...
    @property
    def dim(self) -> int: ...
    @property
    def normalize(self) -> bool: ...
    @property
    def max_length(self) -> int | None: ...
