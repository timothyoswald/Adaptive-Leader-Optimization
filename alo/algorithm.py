from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

BatchObjective = Callable[[np.ndarray], np.ndarray]


@dataclass
class ParticleConfig:
    problem: str = "x2"
    n_particles: int = 200
    n_steps: int = 2000
    dt: float = 0.9
    noise_model: str = "isotropic"
    d: int | None = None
    lambda_accept: float = 1.0
    lambda_reject: float = 1.0

    init_low: float = 7.5
    init_high: float = 10.0
    init_positive_half_axis: bool = True
    reject_negative_proposals: bool = True
    lower_bound: float = 0.0
    upper_bound: float = float("inf")
    seed: int = 7


def _best_index_largest_tie(values: np.ndarray) -> int:
    min_value = np.min(values)
    min_indices = np.flatnonzero(values == min_value)
    return int(min_indices[-1])


def _reject_out_of_bounds_rows(
    proposal: np.ndarray,
    original: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    out_of_bounds = np.any((proposal < lower_bound) | (proposal > upper_bound), axis=1)
    if np.any(out_of_bounds):
        proposal = proposal.copy()
        proposal[out_of_bounds] = original[out_of_bounds]
    return proposal


def _candidate_is_out_of_bounds(candidate: np.ndarray, lower_bound: float, upper_bound: float) -> bool:
    return bool(np.any((candidate < lower_bound) | (candidate > upper_bound)))


def _covariance_sqrt(cov: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def simulate_particles(
    cfg: ParticleConfig,
    dimension: int,
    objective_batch: BatchObjective,
    initial_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]:
    if cfg.n_particles <= 0:
        raise ValueError("n_particles must be > 0")
    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be > 0")
    if cfg.dt <= 0.0:
        raise ValueError("dt must be > 0")
    if cfg.init_low >= cfg.init_high:
        raise ValueError("init_low must be smaller than init_high")
    if cfg.lower_bound >= cfg.upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound")
    if dimension <= 0:
        raise ValueError("dimension must be > 0")
    if cfg.d is not None and cfg.d != dimension:
        raise ValueError(f"cfg.d ({cfg.d}) must match dimension ({dimension}).")
    if cfg.noise_model not in {"isotropic", "anisotropic"}:
        raise ValueError(
            f"noise_model must be 'isotropic' or 'anisotropic', got {cfg.noise_model!r}"
        )

    lambda_accept = float(cfg.lambda_accept)
    lambda_reject = float(cfg.lambda_reject)
    if lambda_accept <= 0.0 or lambda_reject <= 0.0:
        raise ValueError(
            f"lambda_accept and lambda_reject must be > 0 "
            f"(got lambda_accept={lambda_accept}, lambda_reject={lambda_reject})."
        )

    d_eff = int(dimension)

    rng = np.random.default_rng(cfg.seed)
    if initial_positions is None:
        x = rng.uniform(cfg.init_low, cfg.init_high, size=(cfg.n_particles, d_eff))
        if cfg.init_positive_half_axis:
            x = np.abs(x)
    else:
        x = np.asarray(initial_positions, dtype=float).copy()
        if x.shape != (cfg.n_particles, d_eff):
            raise ValueError(
                f"initial_positions must have shape {(cfg.n_particles, d_eff)}, got {x.shape}"
            )

    history = np.zeros((cfg.n_steps + 1, cfg.n_particles, d_eff), dtype=float)
    leader_idx_hist = np.zeros(cfg.n_steps + 1, dtype=int)
    c_hist = np.zeros(cfg.n_steps, dtype=float)

    f_x = objective_batch(x)
    history[0] = x
    leader_idx_hist[0] = _best_index_largest_tie(f_x)

    sqrt_dt = np.sqrt(cfg.dt)
    accept_count = 0
    reject_count = 0
    lambda_n = float(lambda_reject)

    for n in range(cfg.n_steps):
        leader_idx = _best_index_largest_tie(f_x)
        x_best_n = x[leader_idx].copy()
        f_best_n = float(f_x[leader_idx])
        cov_n = np.zeros((d_eff, d_eff), dtype=float)
        cov_sqrt = np.zeros((d_eff, d_eff), dtype=float)
        if cfg.n_particles > 1:
            other_mask = np.ones(cfg.n_particles, dtype=bool)
            other_mask[leader_idx] = False
            diff = x[other_mask] - x_best_n[None, :]
            cov_n = (diff.T @ diff) / float(cfg.n_particles - 1)
            c_sq = float(np.trace(cov_n))
            c_n = float(np.sqrt(max(c_sq, 0.0)))
            if cfg.noise_model == "anisotropic":
                cov_sqrt = _covariance_sqrt(cov_n)
        else:
            other_mask = np.zeros(cfg.n_particles, dtype=bool)
            c_n = 0.0

        x_next = x.copy()
        if cfg.n_particles > 1:
            x_other = x[other_mask]
            noise_other = rng.normal(0.0, 1.0, size=x_other.shape)
            if cfg.noise_model == "anisotropic":
                diffusion_other = sqrt_dt * (noise_other @ cov_sqrt.T)
            else:
                diffusion_other = c_n * sqrt_dt * noise_other
            proposal_other = (
                x_other
                - lambda_n * cfg.dt * (x_other - x_best_n[None, :])
                + diffusion_other
            )
            if cfg.reject_negative_proposals:
                proposal_other = _reject_out_of_bounds_rows(
                    proposal_other,
                    x_other,
                    cfg.lower_bound,
                    cfg.upper_bound,
                )
            x_next[other_mask] = proposal_other

        noise_best = rng.normal(0.0, 1.0, size=(d_eff,))
        if cfg.noise_model == "anisotropic":
            best_candidate = x_best_n + sqrt_dt * (cov_sqrt @ noise_best)
        else:
            best_candidate = x_best_n + c_n * sqrt_dt * noise_best
        if cfg.reject_negative_proposals and _candidate_is_out_of_bounds(
            best_candidate, cfg.lower_bound, cfg.upper_bound
        ):
            accepted = False
            x_best_next = x_best_n
        else:
            f_best_candidate = float(objective_batch(best_candidate[None, :])[0])
            accepted = f_best_candidate < f_best_n
            x_best_next = best_candidate if accepted else x_best_n

        x_next[leader_idx] = x_best_next

        lambda_next = float(lambda_accept if accepted else lambda_reject)
        if accepted:
            accept_count += 1
        else:
            reject_count += 1

        x = x_next
        f_x = objective_batch(x)
        history[n + 1] = x
        leader_idx_hist[n + 1] = _best_index_largest_tie(f_x)
        c_hist[n] = c_n

        lambda_n = lambda_next

    stats: dict[str, int | float] = {
        "accept_count": accept_count,
        "reject_count": reject_count,
        "lambda_accept": lambda_accept,
        "lambda_reject": lambda_reject,
        "lambda_final": lambda_n,
        "noise_model": cfg.noise_model,
    }

    return history, leader_idx_hist, c_hist, stats
