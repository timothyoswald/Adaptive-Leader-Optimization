"""
Compare ALO vs CBO on the four standard 2D benchmarks with consensus-aligned metrics.

Uses softmax consensus m_t ∑ exp(-alpha f_i) x_i / Z applied to BOTH methods (alpha matches CBO).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.cbx_runner import CBXParams, run_cbx_cbo
from common.consensus_metrics import consensus_gap_trajectory, per_run_aggregate_metrics
from common.interface import NoiseModel, stable_seed
from experiments.benchmarks import suite_2d

from alo.algorithm import ParticleConfig, simulate_particles

GAP_TOL = 1e-3


def _init_positions(seed: int, n_particles: int, d: int, lo: float, hi: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.uniform(float(lo), float(hi), size=(int(n_particles), int(d)))


def _method_label(algo: str, noise: NoiseModel) -> str:
    a = "ALO" if algo == "alo" else "CBO"
    n = "isotropic" if noise == "isotropic" else "anisotropic"
    return f"{a} ({n})"


def _global_minimizers(name: str) -> list[np.ndarray]:
    key = name.strip().lower()
    if key == "beale":
        return [np.array([3.0, 0.5], dtype=float)]
    if key == "rosenbrock":
        return [np.array([1.0, 1.0], dtype=float)]
    if key == "rastrigin":
        return [np.array([0.0, 0.0], dtype=float)]
    if key == "himmelblau":
        return [
            np.array([3.0, 2.0], dtype=float),
            np.array([-2.805118, 3.131312], dtype=float),
            np.array([-3.779310, -3.283186], dtype=float),
            np.array([3.584428, -1.848126], dtype=float),
        ]
    raise KeyError(f"Unknown minimizers for benchmark {name!r}")


def _l2_to_nearest_minimizer(x: np.ndarray, mins: list[np.ndarray]) -> float:
    xx = np.asarray(x, dtype=float).reshape(-1)
    return float(min(np.linalg.norm(xx - m.reshape(-1)) for m in mins))


def _particle_variance_over_time(traj: np.ndarray) -> np.ndarray:
    """
    traj: (T+1, N, d). Returns variance time series shape (T+1,).
    Variance definition: mean over coordinates of Var_i[x_{i,j}].
    """
    arr = np.asarray(traj, dtype=float)
    return np.mean(np.var(arr, axis=1, ddof=0), axis=1)


def _one_run_packed(args: tuple[Any, ...]) -> tuple[str, str, NoiseModel, int, float, float, float, float, np.ndarray]:
    cfg, benchmark_name, noise_model, algo, repeat_idx = args

    benchmarks = suite_2d()
    bench = next(b for b in benchmarks if b.name == benchmark_name)
    seed = stable_seed(cfg["seed_base"], "continuous_bench_metrics", benchmark_name, repeat_idx)
    x0 = _init_positions(seed, cfg["n_particles"], bench.dimension, bench.lower_bound, bench.upper_bound)

    if algo == "alo":
        pcfg = ParticleConfig(
            n_particles=int(cfg["n_particles"]),
            n_steps=int(cfg["n_steps"]),
            dt=float(cfg["dt_alo"]),
            noise_model=str(noise_model),
            lambda_accept=float(cfg["lambda_accept"]),
            lambda_reject=float(cfg["lambda_reject"]),
            d=int(bench.dimension),
            seed=int(seed),
            init_low=float(bench.lower_bound),
            init_high=float(bench.upper_bound),
            init_positive_half_axis=False,
            reject_negative_proposals=True,
            lower_bound=float(bench.lower_bound),
            upper_bound=float(bench.upper_bound),
        )
        history, _, _, _stats = simulate_particles(
            cfg=pcfg,
            dimension=int(bench.dimension),
            objective_batch=bench.objective_batch,
            initial_positions=x0,
        )
        traj = np.asarray(history, dtype=float)
    else:
        cbp = CBXParams(**cfg["cbx_params"])  # type: ignore[arg-type]
        _, diag = run_cbx_cbo(
            benchmark=bench,
            seed=int(seed),
            n_particles=int(cfg["n_particles"]),
            n_steps=int(cfg["n_steps"]),
            noise_model=noise_model,
            initial_positions=x0,
            params=cbp,
            attach_particle_history=True,
        )
        traj = np.asarray(diag["particle_history"], dtype=float)

    mets = per_run_aggregate_metrics(
        traj,
        benchmark=bench,
        consensus_alpha=float(cfg["consensus_alpha"]),
        gap_tol=float(cfg["gap_tol"]),
    )
    m_traj, _gaps = consensus_gap_trajectory(traj, benchmark=bench, consensus_alpha=float(cfg["consensus_alpha"]))
    final_consensus = np.asarray(m_traj[-1], dtype=float)
    final_dist = _l2_to_nearest_minimizer(final_consensus, _global_minimizers(bench.name))
    var_t = _particle_variance_over_time(traj)
    return (
        benchmark_name,
        algo,
        noise_model,
        int(repeat_idx),
        mets["consensus_success"],
        mets["first_consensus_step"],
        mets["best_consensus_gap"],
        float(final_dist),
        np.asarray(var_t, dtype=float),
    )


def _aggregate(
    groups: dict[tuple[str, str], list[tuple[float, float, float, float]]]
) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    out: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for key, lst in groups.items():
        srs = np.array([a[0] for a in lst], dtype=float)
        sr_pct = float(100.0 * np.mean(srs))

        steps = np.array([a[1] for a in lst], dtype=float)
        if np.all(np.isnan(steps)):
            med_step = float("nan")
        else:
            med_step = float(np.nanmedian(steps))

        gaps = sorted(float(a[2]) for a in lst)
        med_gap = float(statistics.median(gaps))

        dists = sorted(float(a[3]) for a in lst)
        med_dist = float(statistics.median(dists))

        out[key] = (sr_pct, med_step, med_gap, med_dist)
    return out


def _format_med_step(x: float) -> str:
    if np.isnan(x):
        return "NA"
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.1f}"


def main() -> None:
    mp.freeze_support()

    p = argparse.ArgumentParser(description="Consensus metrics: ALO vs CBO on 2D benchmarks.")
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=90210)
    p.add_argument("--n-particles", type=int, default=200)
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 4) - 1))
    p.add_argument("--out-md", type=str, default="results/four_algorithms_2d_summary.md")
    p.add_argument("--out-csv", type=str, default="results/four_algorithms_2d_runs.csv")
    p.add_argument("--out-variance-plot", type=str, default="results/particle_variance_over_time.png")
    args = p.parse_args()

    cfg: dict[str, Any] = {
        "seed_base": int(args.seed_base),
        "n_particles": int(args.n_particles),
        "n_steps": int(args.n_steps),
        "gap_tol": GAP_TOL,
        "consensus_alpha": 40.0,
        "dt_alo": 0.1361,
        "lambda_accept": 0.2981 / 0.1361,
        "lambda_reject": 0.4324 / 0.1361,
        "cbx_params": {"dt": 0.1, "lamda": 2.5, "sigma": 1.2, "alpha": 40.0},
    }

    benchmarks = suite_2d()
    noises: tuple[NoiseModel, ...] = ("isotropic", "anisotropic")
    algos = ("alo", "cbo")

    tasks = [
        (
            cfg,
            b.name,
            noise_model,
            algo,
            repeat_idx,
        )
        for b in benchmarks
        for noise_model in noises
        for algo in algos
        for repeat_idx in range(int(args.repeats))
    ]

    workers = max(1, int(args.workers))
    if workers == 1:
        rows = [_one_run_packed(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            rows = pool.map(_one_run_packed, tasks, chunksize=4)

    groups: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {}
    var_sums: dict[tuple[str, str], np.ndarray] = {}
    var_counts: dict[tuple[str, str], int] = {}
    csv_lines: list[str] = []

    csv_lines.append("benchmark,method,noise,repeat,success,first_consensus_step,best_consensus_gap,final_consensus_dist")
    for tup in rows:
        bname, algo, nk, repeat_idx, sr, fs, bg, fd, var_t = tup
        key = (bname, _method_label(algo, nk))
        groups.setdefault(key, []).append((sr, fs, bg, fd))
        csv_lines.append(f"{bname},{_method_label(algo, nk)},{nk},{repeat_idx},{sr},{fs},{bg},{fd}")

        kk = (bname, _method_label(algo, nk))
        vt = np.asarray(var_t, dtype=float)
        if kk not in var_sums:
            var_sums[kk] = np.zeros_like(vt)
            var_counts[kk] = 0
        var_sums[kk] += vt
        var_counts[kk] += 1

    agg = _aggregate(groups)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    method_order = [
        _method_label("alo", "isotropic"),
        _method_label("alo", "anisotropic"),
        _method_label("cbo", "isotropic"),
        _method_label("cbo", "anisotropic"),
    ]

    lines: list[str] = []
    lines.append("# Continuous optimization in R^2: ALO vs CBO")
    lines.append("")
    lines.append(
        "Experimental setup: "
        f"particles N={args.n_particles}, steps={args.n_steps}, repeats={args.repeats} "
        "per benchmark and method/noise variant; "
        f"objective gap threshold {GAP_TOL}; softmax consensus with alpha={cfg['consensus_alpha']} (same alpha as CBO). "
        "Median consensus step is the median first time index t (trajectory index, including t=0) where consensus gap <= threshold. "
        "Final distance is ||m_T - x*||_2 (nearest global minimizer if multiple)."
    )
    lines.append("")

    for b in benchmarks:
        lines.append(f"## {b.name.capitalize()}")
        lines.append("")
        lines.append("| Method | Success Rate (SR) % | Median consensus step | Median best consensus gap | Median final ||m_T - x*||_2 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for meth in method_order:
            sr_pc, md_st, gap, dist = agg[(b.name, meth)]
            lines.append(f"| {meth} | {sr_pc:.1f} | {_format_med_step(md_st)} | {gap:.3e} | {dist:.3e} |")
        lines.append("")

    with out_md.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # Variance plots (four separate figures, one per benchmark; 4 curves each).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        _method_label("alo", "isotropic"): "#1f77b4",
        _method_label("alo", "anisotropic"): "#ff7f0e",
        _method_label("cbo", "isotropic"): "#2ca02c",
        _method_label("cbo", "anisotropic"): "#d62728",
    }
    t = np.arange(int(args.n_steps) + 1, dtype=int)

    base_path = Path(args.out_variance_plot)
    out_dir = base_path.parent
    prefix = base_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for b in benchmarks:
        fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.3), constrained_layout=True)
        ax.set_title(f"{b.name.capitalize()} — particle variance over time")
        for meth in method_order:
            kk = (b.name, meth)
            mean_var = var_sums[kk] / max(1, var_counts[kk])
            ax.plot(t, mean_var, label=meth, color=colors[meth], linewidth=1.8)
        ax.grid(True, alpha=0.35)
        ax.set_yscale("log")
        ax.set_ylabel("mean particle variance (log scale)")
        ax.set_xlabel("iteration")
        # Put the color labels inside each plot.
        ax.legend(loc="best", frameon=True, fontsize=9)

        plot_path = out_dir / f"{prefix}_{b.name.lower()}_variance.png"
        fig.savefig(plot_path, dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    main()

