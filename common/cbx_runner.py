from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from common.interface import Benchmark, NoiseModel, RunMetrics, guardrail_diagnostics


@dataclass(frozen=True)
class CBXParams:
    dt: float
    lamda: float
    sigma: float
    alpha: float


def _objective_3d(objective_batch):
    # CBXpy expects an objective that can consume (M, N, d) and return (M, N).
    def f(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.asarray(objective_batch(x), dtype=float)

    return f


def _component_wise_brownian_noise(dyn) -> np.ndarray:
    """
    Component-wise multiplicative noise using the diagonal of the drift.

    Let d = x - c be the drift (particle minus consensus). This returns

        n = sqrt(dt) * diag(d) * z  ==  sqrt(dt) * (d ⊙ z),

    where z ~ N(0, I) i.i.d. per coordinate.

    Note: CBXpy's built-in string mode ``anisotropic`` uses a different scaling; we override it here.
    """
    z = dyn.sampler(size=dyn.drift.shape)
    return np.sqrt(float(dyn.dt)) * (np.asarray(dyn.drift, dtype=float) * np.asarray(z, dtype=float))


def run_cbx_cbo(
    *,
    benchmark: Benchmark,
    seed: int,
    n_particles: int,
    n_steps: int,
    noise_model: NoiseModel,
    initial_positions: np.ndarray,
    params: CBXParams,
    attach_particle_history: bool = False,
) -> tuple[RunMetrics, dict[str, Any]]:
    # Local import so the harness can still import without CBXpy deps installed.
    # Also, `external/CBXpy` is vendored rather than installed, so we add it to sys.path.
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    cbx_root = repo_root / "external" / "CBXpy"
    if str(cbx_root) not in sys.path:
        sys.path.insert(0, str(cbx_root))

    from cbx.dynamics import CBO  # type: ignore

    x0 = np.asarray(initial_positions, dtype=float)
    if x0.shape != (int(n_particles), int(benchmark.dimension)):
        raise ValueError(f"initial_positions must have shape {(n_particles, benchmark.dimension)}, got {x0.shape}")

    f = _objective_3d(benchmark.objective_batch)

    # CBXpy isotropic unchanged; CBXpy's string ``anisotropic`` uses drift ⊗ ξ — use custom noise for √(dt)·N(0,I) axes.
    noise_arg: str | Callable[..., np.ndarray]
    if noise_model == "anisotropic":
        noise_arg = _component_wise_brownian_noise
    elif noise_model == "isotropic":
        noise_arg = "isotropic"
    else:
        raise ValueError(f"Unknown noise_model for CBO: {noise_model!r}")

    dyn = CBO(
        f,
        f_dim="3D",
        x=x0,
        M=1,
        N=int(n_particles),
        d=int(benchmark.dimension),
        max_it=int(n_steps),
        dt=float(params.dt),
        lamda=float(params.lamda),
        sigma=float(params.sigma),
        alpha=float(params.alpha),
        noise=noise_arg,
        seed=int(seed),
        track_args={"names": ["x"], "save_int": 1},
        verbosity=0,
    )

    _best = dyn.optimize()

    x_hist = dyn.history.get("x")
    if not x_hist:
        raise RuntimeError("CBXpy did not record particle history; cannot compute metrics.")

    traj = np.asarray([np.asarray(x[0], dtype=float) for x in x_hist], dtype=float)

    min_each_step = np.array(
        [float(np.min(np.asarray(benchmark.objective_batch(x_t), dtype=float))) for x_t in traj],
        dtype=float,
    )
    best_so_far = np.minimum.accumulate(min_each_step)
    gaps_best = best_so_far - float(benchmark.f_star)

    success_mask = gaps_best <= float(benchmark.success_tol)
    success = int(np.any(success_mask))
    hitting_time = float(np.argmax(success_mask)) if success else float("nan")

    # final_gap: gap of current min at final iteration (not best-so-far).
    final_gap = float(min_each_step[-1] - float(benchmark.f_star))

    metrics = RunMetrics(
        success=success,
        hitting_time=hitting_time,
        best_gap_at_budget=float(gaps_best[-1]),
        final_gap=final_gap,
    )
    diagnostics = guardrail_diagnostics(
        traj,
        min_each_step,
        lower_bound=benchmark.lower_bound,
        upper_bound=benchmark.upper_bound,
    )
    if attach_particle_history:
        diagnostics["particle_history"] = traj
    return metrics, diagnostics

