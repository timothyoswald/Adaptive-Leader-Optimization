from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.cbx_runner import CBXParams, run_cbx_cbo
from common.custom_runner import run_custom
from common.interface import Benchmark, NoiseModel, stable_seed
from experiments.benchmarks import suite_2d, suite_nd


def _init_positions(seed: int, n_particles: int, d: int, lo: float, hi: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.uniform(float(lo), float(hi), size=(int(n_particles), int(d)))


def run_suite(
    *,
    benchmarks: list[Benchmark],
    repeats: int,
    seed_base: int,
    n_particles: int,
    n_steps: int,
    dt: float,
    lambda_accept: float,
    lambda_reject: float,
    cbx_params: CBXParams,
    out_csv: Path | None,
) -> list[dict]:
    rows: list[dict] = []

    for bench in benchmarks:
        for r in range(int(repeats)):
            seed = stable_seed(seed_base, "compare", bench.name, bench.dimension, r)
            x0 = _init_positions(seed, n_particles, bench.dimension, bench.lower_bound, bench.upper_bound)

            for noise in ("isotropic", "anisotropic"):
                noise_model: NoiseModel = noise  # type: ignore[assignment]

                # custom
                m_custom, stats = run_custom(
                    benchmark=bench,
                    seed=seed,
                    n_particles=n_particles,
                    n_steps=n_steps,
                    dt=dt,
                    lambda_accept=lambda_accept,
                    lambda_reject=lambda_reject,
                    noise_model=noise_model,
                    initial_positions=x0,
                )
                rows.append(
                    {
                        "method": "custom",
                        "noise_model": noise,
                        "benchmark": bench.name,
                        "dimension": bench.dimension,
                        "seed": seed,
                        "n_particles": n_particles,
                        "n_steps": n_steps,
                        "dt": dt,
                        "lambda_accept": lambda_accept,
                        "lambda_reject": lambda_reject,
                        **asdict(m_custom),
                        "accept_count": stats.get("accept_count"),
                        "reject_count": stats.get("reject_count"),
                        "diverged": stats.get("diverged"),
                        "nonfinite_state": stats.get("nonfinite_state"),
                        "exploded_state": stats.get("exploded_state"),
                        "boundary_stuck": stats.get("boundary_stuck"),
                        "max_abs_position": stats.get("max_abs_position"),
                        "first_step_boundary_frac": stats.get("first_step_boundary_frac"),
                        "final_boundary_frac": stats.get("final_boundary_frac"),
                    }
                )

                # cbxpy
                m_cbx, cbx_diag = run_cbx_cbo(
                    benchmark=bench,
                    seed=seed,
                    n_particles=n_particles,
                    n_steps=n_steps,
                    noise_model=noise_model,
                    initial_positions=x0,
                    params=cbx_params,
                )
                rows.append(
                    {
                        "method": "cbx_cbo",
                        "noise_model": noise,
                        "benchmark": bench.name,
                        "dimension": bench.dimension,
                        "seed": seed,
                        "n_particles": n_particles,
                        "n_steps": n_steps,
                        "dt": cbx_params.dt,
                        "lamda": cbx_params.lamda,
                        "sigma": cbx_params.sigma,
                        "alpha": cbx_params.alpha,
                        **asdict(m_cbx),
                        **cbx_diag,
                    }
                )

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["2d", "nd"], default="2d")
    p.add_argument("--dimension", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed-base", type=int, default=12345)
    p.add_argument("--n-particles", type=int, default=200)
    p.add_argument("--n-steps", type=int, default=1500)

    p.add_argument("--dt", type=float, default=0.1361)
    p.add_argument("--lambda-accept", type=float, default=0.2981 / 0.1361)
    p.add_argument("--lambda-reject", type=float, default=0.4324 / 0.1361)

    p.add_argument("--cbx-dt", type=float, default=0.1)
    p.add_argument("--cbx-lamda", type=float, default=2.5)
    p.add_argument("--cbx-sigma", type=float, default=1.2)
    p.add_argument("--cbx-alpha", type=float, default=40.0)

    p.add_argument("--out", type=str, default="results/compare.csv")
    args = p.parse_args()

    if args.suite == "2d":
        benchmarks = suite_2d()
    else:
        benchmarks = suite_nd(int(args.dimension))

    cbx_params = CBXParams(dt=args.cbx_dt, lamda=args.cbx_lamda, sigma=args.cbx_sigma, alpha=args.cbx_alpha)
    out_csv = Path(args.out) if args.out else None

    run_suite(
        benchmarks=benchmarks,
        repeats=args.repeats,
        seed_base=args.seed_base,
        n_particles=args.n_particles,
        n_steps=args.n_steps,
        dt=args.dt,
        lambda_accept=args.lambda_accept,
        lambda_reject=args.lambda_reject,
        cbx_params=cbx_params,
        out_csv=out_csv,
    )


if __name__ == "__main__":
    main()

