from __future__ import annotations

import numpy as np

from common.interface import Benchmark


def softmax_consensus_point(x: np.ndarray, energies: np.ndarray, alpha: float) -> np.ndarray:
    """
    Same weighted aggregation as CBXPy CBO consensus: coeffs ∝ exp(-alpha * energy).
    x: shape (N, d), energies: (N,) = f(x_i).
    """
    x = np.asarray(x, dtype=float)
    energies = np.asarray(energies, dtype=float).reshape(-1)
    z = -float(alpha) * energies
    z -= np.max(z)
    w = np.exp(z)
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0.0:
        return np.mean(x, axis=0)
    w /= s
    return np.sum(w[:, None] * x, axis=0)


def per_run_aggregate_metrics(
    history: np.ndarray,
    *,
    benchmark: Benchmark,
    consensus_alpha: float,
    gap_tol: float,
) -> dict[str, float]:
    """
    history: shape (n_steps + 1, N, d) including initialization at index 0.

    Returns floats:
      consensus_success — 1.0 iff min_t consensus_gap(t) <= gap_tol (SR per run indicator)
      first_consensus_step — smallest time index t with consensus_gap(t) <= gap_tol,
                             NaN if never (within horizon)
      best_consensus_gap — min_t (f(m_t) - f*), consensus m_t softmax with consensus_alpha
    """
    traj = np.asarray(history, dtype=float)
    assert traj.ndim == 3
    tol = float(gap_tol)
    gaps_consensus = []

    batch = benchmark.objective_batch
    fs = float(benchmark.f_star)

    for t in range(traj.shape[0]):
        xt = traj[t]
        energies = np.asarray(batch(xt), dtype=float).reshape(-1)

        m = softmax_consensus_point(xt, energies, consensus_alpha)
        f_m = float(np.asarray(batch(m[None, :]), dtype=float).reshape(-1)[0])
        gaps_consensus.append(float(f_m - fs))

    g_c = np.asarray(gaps_consensus, dtype=float)

    success = float(np.nanmin(g_c) <= tol) if np.all(np.isfinite(g_c)) else 0.0

    meets = np.where(g_c <= tol)[0]
    first_consensus_step = float(meets[0]) if meets.size > 0 else float("nan")

    best_gap = float(np.nanmin(g_c))

    return {
        "consensus_success": success,
        "first_consensus_step": first_consensus_step,
        "best_consensus_gap": best_gap,
    }
