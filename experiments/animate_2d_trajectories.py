from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse
import numpy as np

from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alo.algorithm import ParticleConfig, simulate_particles
from experiments.benchmarks import ackley as ackley_fun
from alo.test_fun import beale as beale_fun
from alo.test_fun import himmelblau as himmelblau_fun


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    bounds: tuple[float, float]
    minimizers: list[tuple[float, float]]
    objective_batch: callable


def _batchify(fun):
    def f(x: np.ndarray) -> np.ndarray:
        return np.asarray(fun(x), dtype=float)

    return f


def himmelblau_spec() -> FunctionSpec:
    # Himmelblau has four global minimizers with f=0.
    mins = [
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126),
    ]
    return FunctionSpec(
        name="himmelblau",
        bounds=(-5.0, 5.0),
        minimizers=mins,
        objective_batch=_batchify(himmelblau_fun),
    )


def ackley_spec() -> FunctionSpec:
    return FunctionSpec(
        name="ackley",
        bounds=(-5.0, 5.0),
        minimizers=[(0.0, 0.0)],
        objective_batch=_batchify(ackley_fun),
    )

def beale_spec() -> FunctionSpec:
    # Beale has global minimizer at (3, 0.5) with f=0.
    return FunctionSpec(
        name="beale",
        bounds=(-4.5, 4.5),
        minimizers=[(3.0, 0.5)],
        objective_batch=_batchify(beale_fun),
    )


def _covariance_around_best(x: np.ndarray, leader_idx: int) -> np.ndarray:
    n, d = x.shape
    if n <= 1:
        return np.zeros((d, d), dtype=float)
    mask = np.ones(n, dtype=bool)
    mask[leader_idx] = False
    diff = x[mask] - x[leader_idx][None, :]
    return (diff.T @ diff) / float(n - 1)


def _ellipse_from_cov(cov: np.ndarray, n_std: float = 1.0) -> tuple[float, float, float]:
    # Returns (width, height, angle_degrees)
    if cov.shape != (2, 2):
        return (0.0, 0.0, 0.0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # Axis lengths are 2*n_std*sqrt(eigval) for diameter in each principal axis.
    width = 2.0 * n_std * float(np.sqrt(eigvals[0]))
    height = 2.0 * n_std * float(np.sqrt(eigvals[1]))
    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    return width, height, angle


def _build_contour_grid(spec: FunctionSpec, n: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = spec.bounds
    xs = np.linspace(lo, hi, n)
    ys = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = spec.objective_batch(pts).reshape(X.shape)
    return X, Y, Z


def animate_run(
    *,
    spec: FunctionSpec,
    out_path: Path,
    n_particles: int,
    n_steps: int,
    dt: float,
    lambda_accept: float,
    lambda_reject: float,
    seed: int,
    frame_stride: int,
    interval_ms: int,
    n_std_ellipse: float,
    auto_stop_tol: float | None,
    auto_stop_patience: int,
    min_frames: int,
    hold_last_n_frames: int,
    init_region: str,
) -> None:
    cfg = ParticleConfig(
        n_particles=int(n_particles),
        n_steps=int(n_steps),
        dt=float(dt),
        noise_model="anisotropic",
        lambda_accept=float(lambda_accept),
        lambda_reject=float(lambda_reject),
        d=2,
        seed=int(seed),
        init_low=float(spec.bounds[0]),
        init_high=float(spec.bounds[1]),
        init_positive_half_axis=False,
        reject_negative_proposals=True,
        lower_bound=float(spec.bounds[0]),
        upper_bound=float(spec.bounds[1]),
    )

    # Optional custom initialization region.
    initial_positions = None
    if init_region != "full":
        lo, hi = spec.bounds
        rng = np.random.default_rng(int(seed))
        if init_region == "bottom_left":
            initial_positions = rng.uniform(lo, 0.0, size=(int(n_particles), 2))
        else:
            raise ValueError(f"Unknown init_region={init_region!r}.")

    history, leader_idx_hist, _c_hist, _stats = simulate_particles(
        cfg=cfg,
        dimension=2,
        objective_batch=spec.objective_batch,
        initial_positions=initial_positions,
    )

    # Auto-stop: find first time best-so-far reaches tolerance, then stop shortly after.
    leader_vals_full = spec.objective_batch(history[np.arange(history.shape[0]), leader_idx_hist])
    best_so_far = np.minimum.accumulate(np.asarray(leader_vals_full, dtype=float))
    end_idx = history.shape[0] - 1
    if auto_stop_tol is not None:
        hits = np.flatnonzero(best_so_far <= float(auto_stop_tol))
        if hits.size > 0:
            hit0 = int(hits[0])
            end_idx = min(end_idx, hit0 + int(auto_stop_patience))
    end_idx = max(end_idx, 0)

    # Background contour plot
    X, Y, Z = _build_contour_grid(spec)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    levels = 35
    ax.contourf(X, Y, np.log1p(Z), levels=levels, cmap="viridis", alpha=0.95)
    ax.contour(X, Y, np.log1p(Z), levels=12, colors="k", linewidths=0.3, alpha=0.35)

    # Mark global minimizer(s)
    mins = np.array(spec.minimizers, dtype=float)
    ax.scatter(mins[:, 0], mins[:, 1], marker="*", s=180, c="white", edgecolor="black", linewidth=1.2, zorder=6)
    ax.text(
        mins[0, 0],
        mins[0, 1],
        " global min",
        fontsize=10,
        color="white",
        ha="left",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=2),
        zorder=7,
    )

    # Particle scatter and best-particle marker
    scat = ax.scatter([], [], s=22, c="#A0CBE8", edgecolor="none", alpha=0.9, zorder=4)
    best = ax.scatter([], [], s=110, c="gold", edgecolor="black", linewidth=1.0, zorder=7)

    # Covariance ellipse patch (red)
    ellipse = Ellipse((0, 0), width=0.0, height=0.0, angle=0.0, fill=False, color="red", linewidth=2.0, zorder=8)
    ax.add_patch(ellipse)

    # Title text
    title = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="white",
        bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=3),
        zorder=9,
    )

    lo, hi = spec.bounds
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    frames = list(range(0, history.shape[0], max(1, int(frame_stride))))
    frames = [f for f in frames if f <= end_idx]
    if not frames:
        frames = [0]
    if len(frames) < int(min_frames):
        # Ensure enough frames for a watchable animation (pad with consecutive steps).
        frames = list(range(0, min(end_idx + 1, int(min_frames))))
    if hold_last_n_frames > 0:
        frames = frames + [frames[-1]] * int(hold_last_n_frames)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        best.set_offsets(np.empty((0, 2)))
        ellipse.set_width(0.0)
        ellipse.set_height(0.0)
        title.set_text("")
        return (scat, best, ellipse, title)

    def update(frame_idx: int):
        x = history[frame_idx]
        leader_idx = int(leader_idx_hist[frame_idx])
        x_best = x[leader_idx]
        cov = _covariance_around_best(x, leader_idx)
        w, h, ang = _ellipse_from_cov(cov, n_std=n_std_ellipse)

        scat.set_offsets(x)
        best.set_offsets(x_best[None, :])
        ellipse.center = (float(x_best[0]), float(x_best[1]))
        ellipse.width = w
        ellipse.height = h
        ellipse.angle = ang

        title.set_text(
            f"{spec.name} | n={frame_idx}/{history.shape[0]-1} | N={n_particles} | dt={dt:.4g}\n"
            f"lambda_accept={lambda_accept:.4g}, lambda_reject={lambda_reject:.4g} | ellipse={n_std_ellipse}σ"
        )
        return (scat, best, ellipse, title)

    anim = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=interval_ms)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as GIF by default (most portable).
    # If ffmpeg is installed, users can switch to MP4 by changing extension.
    suffix = out_path.suffix.lower()
    if suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=max(1, int(1000 / max(1, interval_ms))))
        anim.save(out_path, writer=writer, dpi=140)
    else:
        writer = animation.PillowWriter(fps=max(1, int(1000 / max(1, interval_ms))))
        anim.save(out_path, writer=writer, dpi=140)

    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--function", choices=["himmelblau", "ackley", "beale"], required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--N", type=int, default=120)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--dt", type=float, default=0.1361)
    p.add_argument("--lambda-accept", type=float, required=True)
    p.add_argument("--lambda-reject", type=float, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--interval-ms", type=int, default=160)
    p.add_argument("--ellipse-std", type=float, default=1.0)
    p.add_argument("--auto-stop-tol", type=float, default=1e-6)
    p.add_argument("--auto-stop-patience", type=int, default=25)
    p.add_argument("--min-frames", type=int, default=60)
    p.add_argument("--hold-last-n-frames", type=int, default=25)
    p.add_argument("--init-region", choices=["full", "bottom_left"], default="full")
    args = p.parse_args()

    if args.function == "himmelblau":
        spec = himmelblau_spec()
    elif args.function == "ackley":
        spec = ackley_spec()
    else:
        spec = beale_spec()
    animate_run(
        spec=spec,
        out_path=Path(args.out),
        n_particles=args.N,
        n_steps=args.steps,
        dt=args.dt,
        lambda_accept=args.lambda_accept,
        lambda_reject=args.lambda_reject,
        seed=args.seed,
        frame_stride=args.frame_stride,
        interval_ms=args.interval_ms,
        n_std_ellipse=args.ellipse_std,
        auto_stop_tol=(None if args.auto_stop_tol <= 0 else float(args.auto_stop_tol)),
        auto_stop_patience=int(args.auto_stop_patience),
        min_frames=int(args.min_frames),
        hold_last_n_frames=int(args.hold_last_n_frames),
        init_region=str(args.init_region),
    )


if __name__ == "__main__":
    main()

