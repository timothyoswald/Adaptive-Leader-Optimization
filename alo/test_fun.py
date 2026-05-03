"""
Test functions

- Beale (2D only)
- Himmelblau (2D only)
- Rastrigin (nD)
- Rosenbrock (nD)
"""

from __future__ import annotations

from typing import Callable

import numpy as np


Array = np.ndarray
Objective = Callable[[Array], float | Array]


def _to_array(x: Array) -> tuple[Array, bool]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        raise ValueError("Input must have at least one dimension, got scalar.")
    is_single = arr.ndim == 1
    return arr, is_single


def _finalize(y: Array, is_single: bool) -> float | Array:
    return float(y) if is_single else y


def rastrigin(x: Array, a: float = 10.0) -> float | Array:
    """Rastrigin function, supports arbitrary dimension d>=1."""
    arr, is_single = _to_array(x)
    d = arr.shape[-1]
    y = a * d + np.sum(arr * arr - a * np.cos(2.0 * np.pi * arr), axis=-1)
    return _finalize(y, is_single)


def rosenbrock(x: Array) -> float | Array:
    """Rosenbrock function, supports dimension d>=2."""
    arr, is_single = _to_array(x)
    d = arr.shape[-1]
    if d < 2:
        raise ValueError(f"Rosenbrock requires d>=2, got d={d}.")
    x_prev = arr[..., :-1]
    x_next = arr[..., 1:]
    y = np.sum((1.0 - x_prev) ** 2 + 100.0 * (x_next - x_prev * x_prev) ** 2, axis=-1)
    return _finalize(y, is_single)


def beale(x: Array) -> float | Array:
    """Beale function, standard 2D definition only."""
    arr, is_single = _to_array(x)
    d = arr.shape[-1]
    if d != 2:
        raise ValueError(f"Beale is defined here only for d=2, got d={d}.")
    x1 = arr[..., 0]
    x2 = arr[..., 1]
    t1 = 1.5 - x1 + x1 * x2
    t2 = 2.25 - x1 + x1 * (x2**2)
    t3 = 2.625 - x1 + x1 * (x2**3)
    y = t1 * t1 + t2 * t2 + t3 * t3
    return _finalize(y, is_single)


def himmelblau(x: Array) -> float | Array:
    """Himmelblau function, standard 2D definition only."""
    arr, is_single = _to_array(x)
    d = arr.shape[-1]
    if d != 2:
        raise ValueError(f"Himmelblau is defined here only for d=2, got d={d}.")
    x1 = arr[..., 0]
    x2 = arr[..., 1]
    y = (x1 * x1 + x2 - 11.0) ** 2 + (x1 + x2 * x2 - 7.0) ** 2
    return _finalize(y, is_single)


TEST_FUNS: dict[str, Objective] = {
    "beale": beale,
    "himmelblau": himmelblau,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
}


def smoke_test_dims(dims: tuple[int, ...] = (2, 5, 10, 20), seed: int = 0) -> None:
    """Quick dimension smoke test for later direct reuse."""
    rng = np.random.default_rng(seed)

    for d in dims:
        x = rng.normal(size=(7, d))
        _ = rastrigin(x)   # should work in all listed dims
        _ = rosenbrock(x)  # should work in all listed dims (d>=2)
        if d == 2:
            _ = beale(x)
            _ = himmelblau(x)


if __name__ == "__main__":
    smoke_test_dims()
    print("smoke_test_dims passed for d in (2, 5, 10, 20).")
