"""
Sweep particle count N for ALO on a **single 2D benchmark** from `suite_2d()` (default: **Rastrigin**).
All dynamical settings match the plan defaults (alias `experiments/run_compare.py` CLI defaults):

  dt = 0.1361
  lambda_accept = 0.2981 / 0.1361
  lambda_reject  = 0.4324 / 0.1361
  n_steps = 1500
  seed_base = 12345 (with N folded into deterministic seeds)
  repeats = 20 by default (override with --repeats)

Plots **Runtime** vs N and the **mean** (over repeats) of the **best final objective among particles**
(min over the swarm at the last step), for isotropic and anisotropic noise.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.custom_runner import run_custom
from common.interface import NoiseModel, stable_seed
from experiments.benchmarks import suite_2d

# Plan-aligned defaults (do not alter without updating the methodology note above).
_DT = 0.1361
_LAMBDA_ACCEPT = 0.2981 / _DT
_LAMBDA_REJECT = 0.4324 / _DT
_DEFAULT_N_STEPS = 1500
_DEFAULT_SEED_BASE = 12345
_DEFAULT_REPEATS = 20


def _init_positions(seed: int, n_particles: int, d: int, lo: float, hi: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.uniform(float(lo), float(hi), size=(int(n_particles), int(d)))


def _bench_named(name: str):
    key = name.strip().lower()
    for b in suite_2d():
        if b.name.lower() == key:
            return b
    raise RuntimeError(f"Benchmark {name!r} not found in suite_2d().")


def _aggregate_runtime_and_swarm_best(
    rows: list[dict[str, object]],
    *,
    n_vals: list[int],
    noise: str,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (mean_runtime, sem_runtime, mean_swarm_best_obj, sem_swarm_best_obj)."""
    by_n: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        if str(row["noise_model"]) != noise:
            continue
        nkey = int(row["n_particles"])
        by_n.setdefault(nkey, []).append(row)

    mr, sr, mo, so = [], [], [], []
    for n in n_vals:
        grp = by_n.get(n, [])
        times = [float(r["wall_time_s"]) for r in grp]
        objs = [float(r["final_swarm_best_objective"]) for r in grp]
        mr.append(float(np.mean(times)) if times else float("nan"))
        sr.append(float(np.std(times, ddof=1)) / (len(times) ** 0.5) if len(times) > 1 else 0.0)
        mo.append(float(np.mean(objs)) if objs else float("nan"))
        so.append(float(np.std(objs, ddof=1)) / (len(objs) ** 0.5) if len(objs) > 1 else 0.0)
    return mr, sr, mo, so


def main() -> None:
    p = argparse.ArgumentParser(description="ALO sweep over particle count N on one 2D benchmark (plan defaults).")
    p.add_argument(
        "--benchmark",
        type=str,
        default="rastrigin",
        help="Name from suite_2d(), e.g. beale, himmelblau, rastrigin, rosenbrock (default: rastrigin).",
    )
    p.add_argument("--n-min", type=int, default=25)
    p.add_argument("--n-step", type=int, default=25)
    p.add_argument("--n-max", type=int, default=300)
    p.add_argument("--n-steps", type=int, default=_DEFAULT_N_STEPS)
    p.add_argument("--repeats", type=int, default=_DEFAULT_REPEATS)
    p.add_argument("--seed-base", type=int, default=_DEFAULT_SEED_BASE)
    p.add_argument("--out-csv", type=str, default=None)
    p.add_argument("--out-plot", type=str, default=None)
    args = p.parse_args()

    if args.n_step <= 0 or args.n_min <= 0 or args.n_max < args.n_min:
        raise SystemExit("--n-min, --n-step must be positive and --n-max >= --n-min")

    bench = _bench_named(args.benchmark)
    bench_slug = bench.name.lower().replace(" ", "_")
    out_csv_s = args.out_csv or f"results/{bench_slug}_alo_N_sweep.csv"
    out_plot_s = args.out_plot or f"results/{bench_slug}_alo_N_tradeoff.png"
    dt = float(_DT)
    la = float(_LAMBDA_ACCEPT)
    lr = float(_LAMBDA_REJECT)
    n_steps = int(args.n_steps)
    repeats = int(args.repeats)
    seed_base = int(args.seed_base)

    n_list = list(range(int(args.n_min), int(args.n_max) + 1, int(args.n_step)))
    rows: list[dict[str, object]] = []

    for n_particles in n_list:
        for r in range(repeats):
            seed_x = stable_seed(seed_base, "alo_n_sweep_x0", bench.name, n_particles, r)
            seed_run = stable_seed(seed_base, "alo_n_sweep_run", bench.name, n_particles, r)
            x0 = _init_positions(
                seed_x,
                n_particles,
                bench.dimension,
                bench.lower_bound,
                bench.upper_bound,
            )
            for noise in ("isotropic", "anisotropic"):
                nm: NoiseModel = noise  # type: ignore[assignment]
                t0 = time.perf_counter()
                m, stats = run_custom(
                    benchmark=bench,
                    seed=seed_run,
                    n_particles=n_particles,
                    n_steps=n_steps,
                    dt=dt,
                    lambda_accept=la,
                    lambda_reject=lr,
                    noise_model=nm,
                    initial_positions=x0,
                )
                wall = time.perf_counter() - t0
                rows.append(
                    {
                        "benchmark": bench.name,
                        "noise_model": noise,
                        "repeat": r,
                        "n_particles": n_particles,
                        "n_steps": n_steps,
                        "dt": dt,
                        "lambda_accept": la,
                        "lambda_reject": lr,
                        "wall_time_s": wall,
                        "final_swarm_best_objective": stats.get("final_swarm_best_objective"),
                        "success": m.success,
                        "hitting_time": m.hitting_time,
                        "best_gap_at_budget": m.best_gap_at_budget,
                        "final_gap": m.final_gap,
                        "diverged": stats.get("diverged"),
                        "particle_steps_proxy": float(n_particles * n_steps),
                    }
                )

    csv_path = Path(out_csv_s)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {len(rows)} rows to {csv_path.resolve()}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.8), constrained_layout=True, sharex=True)
    colors = {"isotropic": "#1f77b4", "anisotropic": "#ff7f0e"}

    for noise in ("isotropic", "anisotropic"):
        mr, sem_r, mo, sem_o = _aggregate_runtime_and_swarm_best(rows, n_vals=n_list, noise=noise)
        axes[0].errorbar(n_list, mr, yerr=sem_r, fmt="-o", capsize=3, color=colors[noise], label=noise)
        axes[1].errorbar(n_list, mo, yerr=sem_o, fmt="-o", capsize=3, color=colors[noise], label=noise)

    axes[0].set_ylabel("Runtime (s)\n(mean ± SEM over repeats)")
    axes[0].set_title(f"{bench.name.capitalize()} · ALO (plan defaults: dt & λ fixed, steps fixed)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.35)

    axes[1].set_ylabel("Avg. best final objective\n(min over particles at last step)")
    axes[1].set_xlabel(r"particle count $N$")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.35)
    axes[1].set_yscale("symlog", linthresh=1e-12)

    plot_path = Path(out_plot_s)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Wrote plot to {plot_path.resolve()}")


if __name__ == "__main__":
    main()
