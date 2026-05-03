from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.custom_runner import run_custom
from common.interface import NoiseModel, stable_seed
from experiments.benchmarks import suite_2d


@dataclass(frozen=True)
class LambdaRegime:
    name: str
    lambda_accept: float
    lambda_reject: float


def _init_positions(benchmark, seed_x: int, n_particles: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed_x))
    return rng.uniform(float(benchmark.lower_bound), float(benchmark.upper_bound), size=(n_particles, benchmark.dimension))


def _summarize_metrics(metrics: list[dict]) -> tuple[float, float]:
    # Success rate
    sr = float(np.mean([int(m["success"]) for m in metrics])) if metrics else float("nan")
    # Mean hit time over successful runs only.
    hits = [float(m["hitting_time"]) for m in metrics if int(m["success"]) == 1 and not math.isnan(float(m["hitting_time"]))]
    hit_mean = float(np.mean(hits)) if hits else float("nan")
    return sr, hit_mean


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--n-steps", type=int, default=1500)
    p.add_argument("--dt", type=float, default=0.1361)
    p.add_argument("--seed-base", type=int, default=12345)
    p.add_argument("--out-dir", type=str, default="results/ablation_equal_lambda_N50_2d")
    args = p.parse_args()

    benches = suite_2d()
    noises: tuple[NoiseModel, ...] = ("isotropic", "anisotropic")

    # Baseline regime from prior calibration constants (same numeric λ as 0.2981/dt, 0.4324/dt):
    lambda_accept = 0.2981 / float(args.dt)
    lambda_reject = 0.4324 / float(args.dt)
    # Equal regime: set both lambdas equal to the mean of (accept,reject).
    lambda_eq = 0.5 * (lambda_accept + lambda_reject)

    regimes = [
        LambdaRegime("variant1_asymmetric", lambda_accept=lambda_accept, lambda_reject=lambda_reject),
        LambdaRegime("variant2_equal_mean", lambda_accept=lambda_eq, lambda_reject=lambda_eq),
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for bench in benches:
        for r in range(args.repeats):
            seed_x = stable_seed(args.seed_base, "x0", bench.name, bench.dimension, args.N, r)
            x0 = _init_positions(bench, seed_x, args.N)

            for noise in noises:
                seed_run = stable_seed(args.seed_base, "run", bench.name, bench.dimension, args.N, r, noise)
                for reg in regimes:
                    m, stats = run_custom(
                        benchmark=bench,
                        seed=seed_run,
                        n_particles=args.N,
                        n_steps=args.n_steps,
                        dt=args.dt,
                        lambda_accept=reg.lambda_accept,
                        lambda_reject=reg.lambda_reject,
                        noise_model=noise,
                        initial_positions=x0,
                    )

                    rows.append(
                        {
                            "benchmark": bench.name,
                            "dimension": bench.dimension,
                            "N": args.N,
                            "repeat": r,
                            "noise_model": noise,
                            "regime": reg.name,
                            "dt": args.dt,
                            "lambda_accept": reg.lambda_accept,
                            "lambda_reject": reg.lambda_reject,
                            "success": m.success,
                            "hitting_time": m.hitting_time,
                            "best_gap_at_budget": m.best_gap_at_budget,
                            "final_gap": m.final_gap,
                            "diverged": stats.get("diverged", 0),
                        }
                    )

    # Write raw results
    csv_path = out_dir / "raw_results.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # Summarize + plot
    benchmarks_order = [b.name for b in benches]
    regime_names = [reg.name for reg in regimes]
    colors = {"variant1_asymmetric": "#1f77b4", "variant2_equal_mean": "#ff7f0e"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    # Row 0: Success Rate, Row 1: Mean Hit Time
    # Col 0: isotropic, Col 1: anisotropic
    for j, noise in enumerate(noises):
        # Build dicts once per noise
        sr_by_reg = {}
        hit_by_reg = {}
        for regime in regime_names:
            sr_by_reg[regime] = []
            hit_by_reg[regime] = []
            for bname in benchmarks_order:
                rec = [
                    m
                    for m in rows
                    if m["benchmark"] == bname and m["noise_model"] == noise and m["regime"] == regime
                ]
                sr, hit_mean = _summarize_metrics(rec)
                sr_by_reg[regime].append(sr)
                hit_by_reg[regime].append(hit_mean)

        x = np.arange(len(benchmarks_order))
        width = 0.38
        for idx, regime in enumerate(regime_names):
            offs = (-width / 2) if idx == 0 else (width / 2)
            axes[0, j].bar(
                x + offs,
                sr_by_reg[regime],
                width=width,
                color=colors[regime],
                alpha=0.85,
                label=regime if j == 0 else None,
            )
            axes[1, j].bar(
                x + offs,
                hit_by_reg[regime],
                width=width,
                color=colors[regime],
                alpha=0.85,
            )

        axes[0, j].set_title(f"{noise} noise")
        axes[0, j].set_ylabel("Success rate")
        axes[1, j].set_ylabel("Mean hitting time (successful runs)")
        axes[1, j].set_xlabel("Benchmark")
        axes[0, j].set_ylim(0, 1.05)
        axes[0, j].grid(axis="y", alpha=0.25)
        axes[1, j].grid(axis="y", alpha=0.25)
        axes[0, j].set_xticks(x)
        axes[0, j].set_xticklabels(benchmarks_order, rotation=15)
        axes[1, j].set_xticks(x)
        axes[1, j].set_xticklabels(benchmarks_order, rotation=15)

    axes[0, 0].legend(loc="best")
    fig.suptitle(f"2D Ablation at N={args.N}, dt={args.dt} (repeats={args.repeats})", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_plot = out_dir / "ablation_success_and_hittime.png"
    fig.savefig(out_plot, dpi=180)
    print(out_plot)


if __name__ == "__main__":
    main()

