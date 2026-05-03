from __future__ import annotations

import numpy as np

from common.interface import Benchmark, NoiseModel, RunMetrics, guardrail_diagnostics

# ALO optimizer implementation (vendored into this repo).
from alo.algorithm import ParticleConfig, simulate_particles


def _leader_best_so_far(
    history: np.ndarray,
    leader_idx_hist: np.ndarray,
    objective_batch,
    f_star: float,
) -> tuple[np.ndarray, np.ndarray]:
    step_idx = np.arange(history.shape[0])
    leader_positions = history[step_idx, leader_idx_hist]
    leader_vals = np.asarray(objective_batch(leader_positions), dtype=float)
    best_so_far = np.minimum.accumulate(leader_vals)
    gaps_best = best_so_far - float(f_star)
    return leader_vals, gaps_best


def run_custom(
    *,
    benchmark: Benchmark,
    seed: int,
    n_particles: int,
    n_steps: int,
    dt: float,
    lambda_accept: float,
    lambda_reject: float,
    noise_model: NoiseModel,
    initial_positions: np.ndarray,
) -> tuple[RunMetrics, dict[str, int | float]]:
    cfg = ParticleConfig(
        n_particles=int(n_particles),
        n_steps=int(n_steps),
        dt=float(dt),
        noise_model=str(noise_model),
        lambda_accept=float(lambda_accept),
        lambda_reject=float(lambda_reject),
        d=int(benchmark.dimension),
        seed=int(seed),
        init_low=float(benchmark.lower_bound),
        init_high=float(benchmark.upper_bound),
        init_positive_half_axis=False,
        reject_negative_proposals=True,
        lower_bound=float(benchmark.lower_bound),
        upper_bound=float(benchmark.upper_bound),
    )

    history, leader_idx_hist, _c_hist, stats = simulate_particles(
        cfg=cfg,
        dimension=int(benchmark.dimension),
        objective_batch=benchmark.objective_batch,
        initial_positions=np.asarray(initial_positions, dtype=float),
    )

    x_final = np.asarray(history[-1], dtype=float)
    vals_final = np.asarray(benchmark.objective_batch(x_final), dtype=float)
    final_swarm_best = float(np.min(vals_final))

    leader_vals, gaps_best = _leader_best_so_far(
        history, leader_idx_hist, benchmark.objective_batch, benchmark.f_star
    )
    success_mask = gaps_best <= float(benchmark.success_tol)
    success = int(np.any(success_mask))
    hitting_time = float(np.argmax(success_mask)) if success else float("nan")

    final_gap = float(leader_vals[-1] - float(benchmark.f_star))

    metrics = RunMetrics(
        success=success,
        hitting_time=hitting_time,
        best_gap_at_budget=float(gaps_best[-1]),
        final_gap=final_gap,
    )

    diagnostics = guardrail_diagnostics(
        history,
        leader_vals,
        lower_bound=benchmark.lower_bound,
        upper_bound=benchmark.upper_bound,
    )
    stats = dict(stats)
    stats.update(diagnostics)
    stats["final_swarm_best_objective"] = final_swarm_best

    return metrics, stats


# --- import shim ---
