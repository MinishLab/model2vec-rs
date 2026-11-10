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
    ) -> StaticModel: ...
    @staticmethod
    def from_arrays(
        tokenizer: str,
        embeddings: npt.NDArray[np.float32],
        normalize: bool = False,
        weights: npt.NDArray[np.float32] | None = None,
        token_mapping: npt.NDArray[np.int64] | None = None,
    ) -> StaticModel: ...
    @overload
    def encode(self, sentences: str, max_length: int | None = 512, batch_size: int = 1024) -> npt.NDArray[np.float32]: ...
    @overload
    def encode(
        self, sentences: Sequence[str], max_length: int | None = 512, batch_size: int = 1024
    ) -> npt.NDArray[np.float32]: ...
    @property
    def dim(self) -> int: ...
