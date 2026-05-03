from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

BatchObjective = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Benchmark:
    name: str
    dimension: int
    objective_batch: BatchObjective
    f_star: float
    success_tol: float
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class RunMetrics:
    success: int
    hitting_time: float  # NaN if never hits
    best_gap_at_budget: float
    final_gap: float


NoiseModel = Literal["isotropic", "anisotropic"]


def bounds_for_benchmark(name: str) -> tuple[float, float]:
    key = name.strip().lower()
    if key == "rastrigin":
        return (-5.12, 5.12)
    if key == "rosenbrock":
        return (-5.0, 10.0)
    if key == "beale":
        return (-4.5, 4.5)
    if key == "himmelblau":
        return (-5.0, 5.0)
    if key == "ackley":
        return (-5.0, 5.0)
    raise KeyError(f"Unknown benchmark bounds for {name!r}.")


def summarize_best_so_far(best_so_far: np.ndarray, f_star: float, success_tol: float) -> RunMetrics:
    best_so_far = np.asarray(best_so_far, dtype=float)
    gaps = best_so_far - float(f_star)
    success_mask = gaps <= float(success_tol)
    success = int(np.any(success_mask))
    hitting_time = float(np.argmax(success_mask)) if success else float("nan")
    return RunMetrics(
        success=success,
        hitting_time=hitting_time,
        best_gap_at_budget=float(gaps[-1]),
        final_gap=float(gaps[-1]),
    )


def stable_seed(seed_base: int, *parts: object) -> int:
    payload = "|".join([str(int(seed_base)), *(str(p) for p in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32 - 1)


def boundary_hit_fraction(x: np.ndarray, lower_bound: float, upper_bound: float) -> float:
    arr = np.asarray(x, dtype=float)
    scale = max(1.0, abs(float(lower_bound)), abs(float(upper_bound)))
    atol = 1e-9 + 1e-8 * scale
    on_lower = np.isclose(arr, float(lower_bound), atol=atol, rtol=0.0)
    on_upper = np.isclose(arr, float(upper_bound), atol=atol, rtol=0.0)
    return float(np.mean(np.any(on_lower | on_upper, axis=-1)))


def guardrail_diagnostics(
    trajectory: np.ndarray,
    values: np.ndarray,
    *,
    lower_bound: float,
    upper_bound: float,
    explosion_factor: float = 100.0,
) -> dict[str, float | int]:
    traj = np.asarray(trajectory, dtype=float)
    vals = np.asarray(values, dtype=float)
    span = max(1.0, abs(float(lower_bound)), abs(float(upper_bound)))
    max_abs_position = float(np.max(np.abs(traj))) if traj.size else 0.0
    nonfinite_state = int((not np.all(np.isfinite(traj))) or (not np.all(np.isfinite(vals))))
    exploded = int(max_abs_position > explosion_factor * span)

    initial_boundary = boundary_hit_fraction(traj[0], lower_bound, upper_bound) if traj.ndim >= 2 else 0.0
    first_step_boundary = boundary_hit_fraction(traj[1], lower_bound, upper_bound) if traj.shape[0] > 1 else initial_boundary
    final_boundary = boundary_hit_fraction(traj[-1], lower_bound, upper_bound) if traj.ndim >= 2 else 0.0
    boundary_stuck = int(first_step_boundary >= 0.95 or final_boundary >= 0.95)
    diverged = int(nonfinite_state or exploded or boundary_stuck)

    return {
        "nonfinite_state": nonfinite_state,
        "exploded_state": exploded,
        "boundary_stuck": boundary_stuck,
        "diverged": diverged,
        "max_abs_position": max_abs_position,
        "initial_boundary_frac": initial_boundary,
        "first_step_boundary_frac": first_step_boundary,
        "final_boundary_frac": final_boundary,
    }

