"""
Grid scan over (lambda_accept, lambda_reject) with fixed dt and particle count.

Constraints enforced before running:
  - dt < 1
  - 1 - lambda_accept * dt >= 0 and 1 - lambda_reject * dt >= 0
  - lambda_reject >= lambda_accept >= 0

Default run targets **Beale** only with lambda_reject in {0.5, 1.0, 1.5, 2.0} and
lambda_accept in {0.3, 0.8, 1.3, 1.8}, and writes a CSV plus heatmap PNGs when --plot is set.
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.custom_runner import run_custom
from common.interface import Benchmark, NoiseModel, stable_seed
from experiments.benchmarks import suite_2d, suite_nd


def _init_positions(seed: int, n_particles: int, d: int, lo: float, hi: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.uniform(float(lo), float(hi), size=(int(n_particles), int(d)))


def pick_benchmarks(
    *,
    suite: str,
    dimension: int,
    benchmark_names: list[str] | None,
) -> list[Benchmark]:
    if benchmark_names:
        want = {n.strip().lower() for n in benchmark_names}
        pool = suite_2d() if suite == "2d" else suite_nd(int(dimension))
        out = [b for b in pool if b.name.lower() in want]
        missing = want - {b.name.lower() for b in out}
        if missing:
            raise ValueError(f"Unknown or unavailable benchmark(s): {sorted(missing)}")
        return out
    return suite_2d() if suite == "2d" else suite_nd(int(dimension))


def pair_feasible(
    *,
    lambda_accept: float,
    lambda_reject: float,
    dt: float,
    require_dt_lt_one: bool = True,
) -> tuple[bool, str]:
    if require_dt_lt_one and not (0.0 < dt < 1.0):
        return False, f"need 0 < dt < 1, got dt={dt}"
    if lambda_accept < 0 or lambda_reject < 0:
        return False, "lambdas must be >= 0"
    if lambda_accept > lambda_reject:
        return False, "need lambda_accept <= lambda_reject"
    if 1.0 - lambda_accept * dt < -1e-12:
        return False, f"1 - lambda_accept*dt < 0 (got {1.0 - lambda_accept * dt})"
    if 1.0 - lambda_reject * dt < -1e-12:
        return False, f"1 - lambda_reject*dt < 0 (got {1.0 - lambda_reject * dt})"
    return True, ""


def _aggregate_success_matrix(
    rows: list[dict[str, object]],
    *,
    lr_vals: list[float],
    la_vals: list[float],
    noise: str,
    benchmark_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_success_matrix, count_matrix) with shape (len(la_vals), len(lr_vals))."""
    key_fn = lambda r: (
        float(r["lambda_reject"]),
        float(r["lambda_accept"]),
    )
    buckets: dict[tuple[float, float], list[int]] = {}
    for r in rows:
        if str(r["noise_model"]) != noise or str(r["benchmark"]) != benchmark_name:
            continue
        k = key_fn(r)
        buckets.setdefault(k, []).append(int(r["success"]))

    Z = np.full((len(la_vals), len(lr_vals)), np.nan, dtype=float)
    for i, la in enumerate(la_vals):
        for j, lr in enumerate(lr_vals):
            vals = buckets.get((lr, la))
            if vals:
                Z[i, j] = float(np.mean(vals))
    return Z, np.isfinite(Z)


def plot_beale_heatmaps(
    *,
    rows: list[dict[str, object]],
    lr_vals: list[float],
    la_vals: list[float],
    benchmark_name: str,
    plot_path: Path,
    dt: float,
    n_particles: int,
    n_steps: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    vmin, vmax = 0.0, 1.0

    for ax, noise in zip(axes, ("isotropic", "anisotropic")):
        Z, _ = _aggregate_success_matrix(
            rows,
            lr_vals=lr_vals,
            la_vals=la_vals,
            noise=noise,
            benchmark_name=benchmark_name,
        )
        Zm = np.ma.masked_invalid(Z)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="#d9d9d9")
        im = ax.imshow(
            Zm,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            extent=(
                min(lr_vals) - 0.25,
                max(lr_vals) + 0.25,
                min(la_vals) - 0.25,
                max(la_vals) + 0.25,
            ),
        )
        ax.set_xlabel(r"$\lambda_{\mathrm{reject}}$")
        ax.set_ylabel(r"$\lambda_{\mathrm{accept}}$")
        ax.set_title(f"{benchmark_name} | {noise}\nmean success over repeats")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="success rate")

    fig.suptitle(
        f"ALO scan | dt={dt}, N={n_particles}, steps={n_steps}\n"
        r"(blank cells = infeasible constraints or no runs)",
        fontsize=11,
    )
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Grid scan over lambda_accept / lambda_reject (ALO only).")
    p.add_argument("--dt", type=float, default=0.5, help="Fixed time step (must be < 1).")
    p.add_argument("--n-particles", type=int, default=50, help="Particle count N.")
    p.add_argument("--n-steps", type=int, default=1500, help="Optimization horizon.")
    p.add_argument("--repeats", type=int, default=3, help="Repeats per benchmark per lambda pair.")
    p.add_argument("--seed-base", type=int, default=12345)
    p.add_argument(
        "--lambda-reject",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
        help="lambda_reject grid.",
    )
    p.add_argument(
        "--lambda-accept",
        type=float,
        nargs="+",
        default=[0.3, 0.8, 1.3, 1.8],
        help="lambda_accept grid.",
    )
    p.add_argument(
        "--benchmark",
        type=str,
        nargs="+",
        default=["beale"],
        help="Benchmark name(s) from the 2d suite (default: beale only).",
    )
    p.add_argument("--suite", choices=["2d", "nd"], default="2d")
    p.add_argument("--dimension", type=int, default=10)
    p.add_argument("--out", type=str, default="results/beale_lambda_grid_scan.csv")
    p.add_argument(
        "--plot",
        type=str,
        default="results/beale_lambda_heatmap.png",
        help="Write heatmap PNG (mean success vs λ). Empty string disables plotting.",
    )
    args = p.parse_args()

    dt = float(args.dt)
    lr_list = [float(x) for x in args.lambda_reject]
    la_list = [float(x) for x in args.lambda_accept]
    benchmarks = pick_benchmarks(suite=args.suite, dimension=int(args.dimension), benchmark_names=list(args.benchmark))

    rows: list[dict[str, object]] = []
    skipped: list[tuple[float, float, str]] = []

    for lr, la in product(lr_list, la_list):
        ok, reason = pair_feasible(lambda_accept=la, lambda_reject=lr, dt=dt)
        if not ok:
            skipped.append((lr, la, reason))
            continue

        for bench in benchmarks:
            for r in range(int(args.repeats)):
                seed_x = stable_seed(int(args.seed_base), "x0", bench.name, bench.dimension, lr, la, r)
                x0 = _init_positions(
                    seed_x,
                    int(args.n_particles),
                    bench.dimension,
                    bench.lower_bound,
                    bench.upper_bound,
                )
                seed_run = stable_seed(
                    int(args.seed_base), "run", bench.name, bench.dimension, lr, la, r
                )
                for noise in ("isotropic", "anisotropic"):
                    nm: NoiseModel = noise  # type: ignore[assignment]
                    m, stats = run_custom(
                        benchmark=bench,
                        seed=seed_run,
                        n_particles=int(args.n_particles),
                        n_steps=int(args.n_steps),
                        dt=dt,
                        lambda_accept=la,
                        lambda_reject=lr,
                        noise_model=nm,
                        initial_positions=x0,
                    )
                    rows.append(
                        {
                            "benchmark": bench.name,
                            "noise_model": noise,
                            "repeat": r,
                            "dt": dt,
                            "lambda_accept": la,
                            "lambda_reject": lr,
                            "one_minus_la_dt": 1.0 - la * dt,
                            "one_minus_lr_dt": 1.0 - lr * dt,
                            "success": m.success,
                            "hitting_time": m.hitting_time,
                            "best_gap_at_budget": m.best_gap_at_budget,
                            "final_gap": m.final_gap,
                            "diverged": stats.get("diverged"),
                            "nonfinite_state": stats.get("nonfinite_state"),
                            "boundary_stuck": stats.get("boundary_stuck"),
                        }
                    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Wrote {len(rows)} rows to {out_path.resolve()}")
    if skipped:
        print(f"Skipped {len(skipped)} (lambda_reject, lambda_accept) pairs:")
        for lr, la, reason in skipped[:25]:
            print(f"  lr={lr}, la={la}: {reason}")
        if len(skipped) > 25:
            print(f"  ... and {len(skipped) - 25} more")

    plot_arg = (args.plot or "").strip()
    if plot_arg and len(benchmarks) == 1:
        plot_beale_heatmaps(
            rows=rows,
            lr_vals=lr_list,
            la_vals=la_list,
            benchmark_name=benchmarks[0].name,
            plot_path=Path(plot_arg),
            dt=dt,
            n_particles=int(args.n_particles),
            n_steps=int(args.n_steps),
        )
        print(f"Wrote heatmap to {Path(plot_arg).resolve()}")
    elif plot_arg and len(benchmarks) != 1:
        print("Skipping plot: use exactly one --benchmark for the heatmap export.")


if __name__ == "__main__":
    main()
