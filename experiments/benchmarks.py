from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from common.interface import Benchmark, bounds_for_benchmark

from alo.test_fun import beale, himmelblau, rastrigin, rosenbrock


BatchObjective = Callable[[np.ndarray], np.ndarray]


def _batchify(fun: Callable[[np.ndarray], float | np.ndarray]) -> BatchObjective:
    def f(x: np.ndarray) -> np.ndarray:
        return np.asarray(fun(x), dtype=float)

    return f


def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2.0 * np.pi) -> float | np.ndarray:
    """Ackley function, supports arbitrary dimension d>=1."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
        single = True
    else:
        single = False
    d = arr.shape[-1]
    if d <= 0:
        raise ValueError("Ackley requires d>=1.")
    x_sq = np.mean(arr * arr, axis=-1)
    x_cos = np.mean(np.cos(c * arr), axis=-1)
    y = -a * np.exp(-b * np.sqrt(x_sq)) - np.exp(x_cos) + a + np.e
    return float(y[0]) if single else y


def suite_2d() -> list[Benchmark]:
    out: list[Benchmark] = []
    for name, fun, tol in [
        ("beale", beale, 1e-10),
        ("himmelblau", himmelblau, 1e-10),
        ("rastrigin", rastrigin, 1e-10),
        ("rosenbrock", rosenbrock, 1e-10),
    ]:
        lo, hi = bounds_for_benchmark(name)
        out.append(
            Benchmark(
                name=name,
                dimension=2,
                objective_batch=_batchify(fun),
                f_star=0.0,
                success_tol=tol,
                lower_bound=lo,
                upper_bound=hi,
            )
        )
    return out


def suite_nd(dimension: int) -> list[Benchmark]:
    if dimension < 2:
        raise ValueError("suite_nd expects dimension>=2.")
    out: list[Benchmark] = []
    # Higher-dimensional “success” is typically evaluated with a looser tolerance.
    # The paper’s comparative tables emphasize gap <= 1e-3 (SR90); we default to 1e-3 here.
    tol = 1e-3
    for name, fun, tol in [
        ("rastrigin", rastrigin, tol),
        ("rosenbrock", rosenbrock, tol),
        ("ackley", ackley, tol),
    ]:
        lo, hi = bounds_for_benchmark(name)
        out.append(
            Benchmark(
                name=name,
                dimension=int(dimension),
                objective_batch=_batchify(fun),
                f_star=0.0,
                success_tol=tol,
                lower_bound=lo,
                upper_bound=hi,
            )
        )
    return out

