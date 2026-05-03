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
from common.consensus_metrics import per_run_aggregate_metrics
from common.interface import NoiseModel, stable_seed
from experiments.benchmarks import suite_2d

from external.__21_410_import import ParticleConfig, simulate_particles  # type: ignore[import-not-found]

GAP_TOL = 1e-3


def _init_positions(seed: int, n_particles: int, d: int, lo: float, hi: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.uniform(float(lo), float(hi), size=(int(n_particles), int(d)))


def _method_label(algo: str, noise: NoiseModel) -> str:
    a = "ALO" if algo == "alo" else "CBO"
    n = "isotropic" if noise == "isotropic" else "anisotropic"
    return f"{a} ({n})"


def _one_run_packed(args: tuple[Any, ...]) -> tuple[str, str, NoiseModel, int, float, float, float]:
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
    return (
        benchmark_name,
        algo,
        noise_model,
        int(repeat_idx),
        mets["consensus_success"],
        mets["first_consensus_step"],
        mets["best_consensus_gap"],
    )


def _aggregate(groups: dict[tuple[str, str], list[tuple[float, float, float]]]) -> dict[tuple[str, str], tuple[float, float, float]]:
    out: dict[tuple[str, str], tuple[float, float, float]] = {}
    for key, lst in groups.items():
        srs = np.array([a[0] for a in lst], dtype=float)
        sr_pct = float(100.0 * np.mean(srs))

        steps = np.array([a[1] for a in lst], dtype=float)
        if np.all(np.isnan(steps)):
            med_step = float("nan")
        else:
            med_step = float(np.nanmedian(steps))

        finals = sorted(float(a[2]) for a in lst)
        med_final = float(statistics.median(finals))

        out[key] = (sr_pct, med_step, med_final)
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
    p.add_argument("--n-steps", type=int, default=1500)
    p.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 4) - 1))
    p.add_argument("--out-md", type=str, default="results/continuous_benchmark_table.md")
    p.add_argument("--out-csv", type=str, default="results/continuous_benchmark_runs.csv")
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

    groups: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    csv_lines: list[str] = []

    csv_lines.append("benchmark,method,noise,repeat,success,first_consensus_step,best_consensus_gap")
    for tup in rows:
        bname, algo, nk, repeat_idx, sr, fs, bg = tup
        key = (bname, _method_label(algo, nk))
        groups.setdefault(key, []).append((sr, fs, bg))
        csv_lines.append(f"{bname},{_method_label(algo, nk)},{nk},{repeat_idx},{sr},{fs},{bg}")

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
    lines.append("# Continuous optimization: consensus metrics")
    lines.append("")
    lines.append(
        "Experimental setup: "
        f"particles N={args.n_particles}, steps={args.n_steps}, repeats={args.repeats} "
        "per benchmark and method/noise variant; "
        f"objective gap threshold {GAP_TOL}; softmax consensus with alpha={cfg['consensus_alpha']} (same alpha as CBO). "
        "Median step is the median first time index t (trajectory index, including t=0) where consensus gap <= threshold."
    )
    lines.append("")

    for b in benchmarks:
        lines.append(f"## {b.name.capitalize()}")
        lines.append("")
        lines.append("| Method | Success Rate (SR) % | Median consensus step | Final objective value |")
        lines.append("| --- | ---: | ---: | ---: |")
        for meth in method_order:
            sr_pc, md_st, fn = agg[(b.name, meth)]
            lines.append(f"| {meth} | {sr_pc:.1f} | {_format_med_step(md_st)} | {fn:.3e} |")
        lines.append("")

    with out_md.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()

