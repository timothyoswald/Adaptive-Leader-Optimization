"""
Rosenbrock experiment: far-from-minimum initialization in a fixed box.

Outputs (by default under results/rosenbrock_far_init/):
  1) rosenbrock_init_region.png — surface/contours + init rectangle + example scatter
  2) best_so_far_gap_median.png — median best-so-far objective gap vs time (4 methods)
  3) consensus_vs_best_gap_median.png — median consensus gap vs median best-so-far gap (per method)
  4) explore_vs_collapse_median.png — median ||m_t - x*||^2 vs median mean-squared spread around m_t

Does not run automatically; execute as a script.

Consensus in *all* panels is the **softmax aggregate** used elsewhere in this repo:
  m_t = sum_i x_i exp(-alpha f(x_i)) / sum_i exp(-alpha f(x_i)),
and **consensus gap** means f(m_t) - f* (evaluated at that point). This is **not**
ALO's internal "leader / best particle" unless alpha is extremely large — it is chosen
for a fair comparison to CBO's consensus notion.

Objective-gap plots use a **display floor** (``--plot-gap-floor``) so one method reaching
almost-exact zeros does not squash the rest on a log scale (raw trajectory data unchanged).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.cbx_runner import CBXParams, run_cbx_cbo
from common.consensus_metrics import consensus_gap_trajectory
from common.interface import NoiseModel, stable_seed
from experiments.benchmarks import suite_2d

from alo.algorithm import ParticleConfig, simulate_particles


def _init_in_box(seed: int, n_particles: int, x1_lo: float, x1_hi: float, x2_lo: float, x2_hi: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    x1 = rng.uniform(float(x1_lo), float(x1_hi), size=(int(n_particles),))
    x2 = rng.uniform(float(x2_lo), float(x2_hi), size=(int(n_particles),))
    return np.stack([x1, x2], axis=1)


def _best_so_far_gap(traj: np.ndarray, objective_batch, f_star: float) -> np.ndarray:
    """
    traj: (T+1, N, d)
    returns g_t = min_{s<=t} min_i f(x_i,s) - f*
    """
    mins: List[float] = []
    for t in range(int(traj.shape[0])):
        v = float(np.min(np.asarray(objective_batch(traj[t]), dtype=float)))
        mins.append(v)
    m = np.minimum.accumulate(np.asarray(mins, dtype=float))
    return m - float(f_star)


def _mean_sq_spread_around_consensus(traj: np.ndarray, m_traj: np.ndarray) -> np.ndarray:
    """
    traj: (T+1, N, d), m_traj: (T+1, d)
    returns s_t = mean_i ||x_{i,t} - m_t||^2
    """
    x = np.asarray(traj, dtype=float)
    m = np.asarray(m_traj, dtype=float)
    diff = x - m[:, None, :]
    return np.mean(np.sum(diff * diff, axis=-1), axis=-1)


def _dist2_to_minimum(m_traj: np.ndarray, x_star: np.ndarray) -> np.ndarray:
    d = m_traj - np.asarray(x_star, dtype=float).reshape(1, -1)
    return np.sum(d * d, axis=-1)


def _method_label(algo: str, noise: NoiseModel) -> str:
    a = "ALO" if algo == "alo" else "CBO"
    n = "isotropic" if noise == "isotropic" else "anisotropic"
    return f"{a} ({n})"


def _one_run(args: Tuple[Any, ...]) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg, noise_model, algo, repeat_idx = args

    benchmarks = suite_2d()
    bench = next(b for b in benchmarks if b.name == "rosenbrock")
    seed = stable_seed(cfg["seed_base"], "rosenbrock_far_init", algo, noise_model, repeat_idx)

    x0 = _init_in_box(
        seed,
        cfg["n_particles"],
        cfg["x1_lo"],
        cfg["x1_hi"],
        cfg["x2_lo"],
        cfg["x2_hi"],
    )

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

    m_traj, g_cons = consensus_gap_trajectory(traj, benchmark=bench, consensus_alpha=float(cfg["consensus_alpha"]))
    g_best = _best_so_far_gap(traj, bench.objective_batch, bench.f_star)
    spread = _mean_sq_spread_around_consensus(traj, m_traj)
    dmin = _dist2_to_minimum(m_traj, cfg["x_star"])

    label = _method_label(algo, noise_model)
    return label, g_cons, g_best, spread, dmin


def _plot_init_figure(
    *,
    out_path: Path,
    objective_batch,
    x_star: np.ndarray,
    x1_lo: float,
    x1_hi: float,
    x2_lo: float,
    x2_hi: float,
    example_x0: np.ndarray,
    grid_n: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_star = np.asarray(x_star, dtype=float).reshape(-1)

    # Domain: include init box and global minimizer comfortably.
    pad = 0.35
    x1a = min(float(x1_lo), float(x_star[0])) - pad
    x1b = max(float(x1_hi), float(x_star[0])) + pad
    x2a = min(float(x2_lo), float(x_star[1])) - pad
    x2b = max(float(x2_hi), float(x_star[1])) + pad

    g1 = np.linspace(x1a, x1b, int(grid_n))
    g2 = np.linspace(x2a, x2b, int(grid_n))
    G1, G2 = np.meshgrid(g1, g2, indexing="xy")
    pts = np.stack([G1.ravel(), G2.ravel()], axis=1)
    Z = np.asarray(objective_batch(pts), dtype=float).reshape(G1.shape)

    fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.8), constrained_layout=True)
    # log-coloring helps visualize the Rosenbrock valley.
    Zp = np.maximum(Z, 1e-12)
    cf = ax.contourf(G1, G2, np.log10(Zp), levels=40, cmap="viridis")
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(r"$\log_{10}(f(x))$")

    ax.plot([x1_lo, x1_hi, x1_hi, x1_lo, x1_lo], [x2_lo, x2_lo, x2_hi, x2_hi, x2_lo], color="white", linewidth=2.2)
    ax.scatter(example_x0[:, 0], example_x0[:, 1], s=10, c="white", alpha=0.55, linewidths=0)
    ax.scatter([float(x_star[0])], [float(x_star[1])], s=120, marker="*", c="red", edgecolors="black", linewidths=0.6, zorder=5)

    ax.set_title("Rosenbrock: initialization region (white box) + example particles")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal", adjustable="box")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    mp.freeze_support()

    p = argparse.ArgumentParser(description="Rosenbrock far-init experiment: diagnostics plots.")
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=90421)
    p.add_argument("--n-particles", type=int, default=200)
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 4) - 1))

    p.add_argument("--x1-lo", type=float, default=-1.5)
    p.add_argument("--x1-hi", type=float, default=-0.5)
    p.add_argument("--x2-lo", type=float, default=1.5)
    p.add_argument("--x2-hi", type=float, default=2.5)

    p.add_argument("--consensus-alpha", type=float, default=40.0)

    p.add_argument("--out-dir", type=str, default=str(_REPO_ROOT / "results" / "rosenbrock_far_init"))

    # Optional: nicer init visualization grid
    p.add_argument("--init-grid-n", type=int, default=220)

    # Log-scale readability: smallest gap shown on objective-gap figures (figures 2–3).
    p.add_argument(
        "--plot-gap-floor",
        type=float,
        default=1e-12,
        help="Clip objective gaps below this positive value ONLY when plotting log-y (default: 1e-12).",
    )

    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = suite_2d()
    bench = next(b for b in benchmarks if b.name == "rosenbrock")
    x_star = np.array([1.0, 1.0], dtype=float)

    cfg: Dict[str, Any] = {
        "seed_base": int(args.seed_base),
        "n_particles": int(args.n_particles),
        "n_steps": int(args.n_steps),
        "consensus_alpha": float(args.consensus_alpha),
        # match the continuous benchmark harness defaults unless overridden later
        "dt_alo": 0.1361,
        "lambda_accept": 0.2981 / 0.1361,
        "lambda_reject": 0.4324 / 0.1361,
        "cbx_params": {"dt": 0.1, "lamda": 2.5, "sigma": 1.2, "alpha": float(args.consensus_alpha)},
        "x1_lo": float(args.x1_lo),
        "x1_hi": float(args.x1_hi),
        "x2_lo": float(args.x2_lo),
        "x2_hi": float(args.x2_hi),
        "x_star": x_star,
    }

    # Figure 1: init visualization (use repeat 0, ALO isotropic stable seed ordering doesn't matter much)
    demo_seed = stable_seed(cfg["seed_base"], "rosenbrock_far_init", "alo", "isotropic", 0)
    example_x0 = _init_in_box(
        demo_seed,
        int(args.n_particles),
        cfg["x1_lo"],
        cfg["x1_hi"],
        cfg["x2_lo"],
        cfg["x2_hi"],
    )
    _plot_init_figure(
        out_path=out_dir / "rosenbrock_init_region.png",
        objective_batch=bench.objective_batch,
        x_star=x_star,
        x1_lo=cfg["x1_lo"],
        x1_hi=cfg["x1_hi"],
        x2_lo=cfg["x2_lo"],
        x2_hi=cfg["x2_hi"],
        example_x0=example_x0,
        grid_n=int(args.init_grid_n),
    )

    noises: Tuple[NoiseModel, ...] = ("isotropic", "anisotropic")
    algos = ("alo", "cbo")

    tasks = [(cfg, nk, algo, repeat_idx) for nk in noises for algo in algos for repeat_idx in range(int(args.repeats))]
    workers = max(1, int(args.workers))
    if workers == 1:
        rows = [_one_run(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            rows = pool.map(_one_run, tasks, chunksize=4)

    method_order = [
        _method_label("alo", "isotropic"),
        _method_label("alo", "anisotropic"),
        _method_label("cbo", "isotropic"),
        _method_label("cbo", "anisotropic"),
    ]

    series: Dict[str, Dict[str, List[np.ndarray]]] = {
        m: {"g_cons": [], "g_best": [], "spread": [], "dmin": []} for m in method_order
    }

    for label, g_cons, g_best, spread, dmin in rows:
        bucket = series[label]
        bucket["g_cons"].append(np.asarray(g_cons, dtype=float))
        bucket["g_best"].append(np.asarray(g_best, dtype=float))
        bucket["spread"].append(np.asarray(spread, dtype=float))
        bucket["dmin"].append(np.asarray(dmin, dtype=float))

    medians: Dict[str, Dict[str, np.ndarray]] = {}
    for m in method_order:
        medians[m] = {
            "g_cons": np.nanmedian(np.stack(series[m]["g_cons"], axis=0), axis=0),
            "g_best": np.nanmedian(np.stack(series[m]["g_best"], axis=0), axis=0),
            "spread": np.nanmedian(np.stack(series[m]["spread"], axis=0), axis=0),
            "dmin": np.nanmedian(np.stack(series[m]["dmin"], axis=0), axis=0),
        }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(int(args.n_steps) + 1, dtype=int)
    gap_floor = float(np.nextafter(max(float(args.plot_gap_floor), 1e-323), np.inf))

    colors = {
        _method_label("alo", "isotropic"): "#1f77b4",
        _method_label("alo", "anisotropic"): "#ff7f0e",
        _method_label("cbo", "isotropic"): "#2ca02c",
        _method_label("cbo", "anisotropic"): "#d62728",
    }

    # Figure 2: median best-so-far objective gap
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.4), constrained_layout=True)
    for m in method_order:
        ax.plot(t, np.maximum(medians[m]["g_best"], gap_floor), label=m, color=colors[m], linewidth=2.0)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(f"Median best-so-far objective gap (log-y clipped at {gap_floor:g})")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\min_{s\leq t}\min_i f(x_{i,s}) - f^\star$ (log scale)")
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.savefig(out_dir / "best_so_far_gap_median.png", dpi=170)
    plt.close(fig)

    # Figure 3: consensus vs best particle gaps (medians)
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 5.4), constrained_layout=True)
    for m in method_order:
        ax.plot(
            t,
            np.maximum(medians[m]["g_cons"], gap_floor),
            linestyle="--",
            linewidth=2.0,
            color=colors[m],
            label=f"{m}: consensus",
        )
        ax.plot(
            t,
            np.maximum(medians[m]["g_best"], gap_floor),
            linestyle="-",
            linewidth=1.9,
            color=colors[m],
            alpha=0.95,
            label=f"{m}: best-so-far",
        )
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(
        f"Consensus gap vs best-so-far gap (median trials; log-y clipped at {gap_floor:g})"
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"objective gap (log scale): $f(m_t)-f^\star$ vs best-so-far")
    ax.legend(ncol=2, fontsize=8, frameon=True, loc="upper right")
    fig.savefig(out_dir / "consensus_vs_best_gap_median.png", dpi=170)
    plt.close(fig)

    # Figure 4: explore vs collapse wrong (medians)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.8), constrained_layout=True)
    pairs = [
        ("ALO (isotropic)", _method_label("alo", "isotropic")),
        ("ALO (anisotropic)", _method_label("alo", "anisotropic")),
        ("CBO (isotropic)", _method_label("cbo", "isotropic")),
        ("CBO (anisotropic)", _method_label("cbo", "anisotropic")),
    ]
    for ax, (ttl, key) in zip(np.ravel(axes), pairs):
        ax2 = ax.twinx()
        (l1,) = ax.plot(
            t,
            medians[key]["spread"],
            color="#9467bd",
            linewidth=2.1,
            label=r"spread: $\mathbb{E}_i\|x_i-m_t\|^2$ (median trial)",
        )
        (l2,) = ax2.plot(
            t,
            np.maximum(medians[key]["dmin"], 1e-300),
            color="#17becf",
            linewidth=2.1,
            linestyle="--",
        )
        ax2.set_yscale("log")
        ax.set_title(ttl)
        ax.set_xlabel("iteration")
        ax.set_ylabel("mean squared spread around consensus")
        ax2.set_ylabel(r"$\|m_t-x^\star\|_2^2$ (log scale)")
        ax.grid(True, alpha=0.3)
        ax.legend(handles=[l1], loc="upper left", fontsize=9)
        ax2.legend([l2], [r"consensus $\|m_t-x^\star\|^2$ (median trial)"], loc="upper right", fontsize=9)

    fig.suptitle("Explore vs premature collapse diagnostics (medians across trials)", fontsize=12)
    fig.savefig(out_dir / "explore_vs_collapse_median.png", dpi=170)
    plt.close(fig)

    written = [
        out_dir / "rosenbrock_init_region.png",
        out_dir / "best_so_far_gap_median.png",
        out_dir / "consensus_vs_best_gap_median.png",
        out_dir / "explore_vs_collapse_median.png",
    ]
    print("Wrote figures to:", str(out_dir))
    for wp in written:
        print(" ", wp)


if __name__ == "__main__":
    main()
