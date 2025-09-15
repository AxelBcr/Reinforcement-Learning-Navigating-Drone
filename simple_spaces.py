# iMPORTS
from __future__ import annotations
from typing import Iterable, Sequence, Tuple as TupleType, Any
import numpy as np


class Box:
    """Continuous space that is bounded by ``low`` and ``high`` arrays."""

    def __init__(self, low: Sequence[float], high: Sequence[float], dtype=np.float32):
        self.dtype = np.dtype(dtype)
        self.low = np.asarray(low, dtype=self.dtype)
        self.high = np.asarray(high, dtype=self.dtype)

        if self.low.shape != self.high.shape:
            raise ValueError("`low` and `high` must share the same shape.")
        if np.any(self.high < self.low):
            raise ValueError("All entries of `high` must be greater than or equal to `low`.")

    @property
    def shape(self) -> TupleType[int, ...]:
        return self.low.shape

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return a random sample from the box."""
        rng = rng or np.random.default_rng()
        return rng.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x: Sequence[float]) -> bool:
        """Check whether ``x`` lies inside the box bounds."""
        arr = np.asarray(x, dtype=self.dtype)
        if arr.shape != self.shape:
            return False
        return np.all(arr >= self.low) and np.all(arr <= self.high)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Box(low={self.low}, high={self.high}, dtype={self.dtype})"


class Discrete:
    """Finite space of integers in ``[0, n)``."""

    def __init__(self, n: int):
        n_int = int(n)
        if n_int <= 0:
            raise ValueError("`n` must be a positive integer.")
        self.n = n_int

    def sample(self, rng: np.random.Generator | None = None) -> int:
        rng = rng or np.random.default_rng()
        return int(rng.integers(self.n))

    def contains(self, x: Any) -> bool:
        try:
            value = int(x)
        except (TypeError, ValueError):
            return False
        return 0 <= value < self.n

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Discrete(n={self.n})"


class Tuple:
    """Cartesian product of multiple independent spaces."""

    def __init__(self, spaces: Iterable[Any]):
        self.spaces: TupleType[Any, ...] = tuple(spaces)
        if not self.spaces:
            raise ValueError("`spaces` must contain at least one element.")

    def sample(self, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        return tuple(space.sample(rng) if hasattr(space, "sample") else None for space in self.spaces)

    def contains(self, x: Sequence[Any]) -> bool:
        try:
            iterator = iter(x)
        except TypeError:
            return False
        values = list(iterator)
        if len(values) != len(self.spaces):
            return False
        for space, value in zip(self.spaces, values):
            if hasattr(space, "contains") and not space.contains(value):
                return False
        return True

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Tuple(spaces={self.spaces!r})"


class _SpacesNamespace:
    """Namespace providing an API similar to ``gym.spaces``."""

    Box = Box
    Discrete = Discrete
    Tuple = Tuple


spaces = _SpacesNamespace()


__all__ = ["Box", "Discrete", "Tuple", "spaces"]