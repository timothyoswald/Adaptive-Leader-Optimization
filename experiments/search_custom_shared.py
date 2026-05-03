from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Candidate:
    dt: float
    lambda_accept: float
    lambda_reject: float


def _init_positions(seed: int, n_particles: int, d: int, lo: float, hi: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.uniform(float(lo), float(hi), size=(int(n_particles), int(d)))


def _evaluate_candidate(
    cand: Candidate,
    benchmarks: list[Benchmark],
    *,
    repeats: int,
    seed_base: int,
    n_particles: int,
    n_steps: int,
) -> list[dict]:
    rows: list[dict] = []

    for bench in benchmarks:
        for r in range(int(repeats)):
            seed = stable_seed(seed_base, "search", bench.name, bench.dimension, r)
            x0 = _init_positions(seed, n_particles, bench.dimension, bench.lower_bound, bench.upper_bound)
            for noise in ("isotropic", "anisotropic"):
                nm: NoiseModel = noise  # type: ignore[assignment]
                m, stats = run_custom(
                    benchmark=bench,
                    seed=seed,
                    n_particles=n_particles,
                    n_steps=n_steps,
                    dt=cand.dt,
                    lambda_accept=cand.lambda_accept,
                    lambda_reject=cand.lambda_reject,
                    noise_model=nm,
                    initial_positions=x0,
                )
                rows.append(
                    {
                        "benchmark": bench.name,
                        "noise_model": noise,
                        "seed": seed,
                        "success": m.success,
                        "hitting_time": m.hitting_time,
                        "best_gap_at_budget": m.best_gap_at_budget,
                        "final_gap": m.final_gap,
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

    return rows


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}

    # group by (benchmark, noise_model)
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((str(r["benchmark"]), str(r["noise_model"])), []).append(r)

    per_bn: list[dict] = []
    for (bench, noise), recs in groups.items():
        success_arr = np.array([int(rr["success"]) for rr in recs], dtype=int)
        sr_hat = float(np.mean(success_arr)) if len(success_arr) else math.nan

        hits = np.array([float(rr["hitting_time"]) for rr in recs if int(rr["success"]) == 1], dtype=float)
        hit_mean = float(np.mean(hits)) if hits.size else math.nan
        hit_med = float(np.median(hits)) if hits.size else math.nan

        gaps = np.array([float(rr["best_gap_at_budget"]) for rr in recs], dtype=float)
        gap_mean = float(np.mean(gaps)) if gaps.size else math.nan
        diverged_frac = float(np.mean([int(rr.get("diverged", 0)) for rr in recs])) if recs else math.nan

        per_bn.append(
            {
                "benchmark": bench,
                "noise_model": noise,
                "sr_hat": sr_hat,
                "hit_mean_success": hit_mean,
                "hit_median_success": hit_med,
                "best_gap_mean": gap_mean,
                "diverged_frac": diverged_frac,
            }
        )

    sr_vals = np.array([float(r["sr_hat"]) for r in per_bn], dtype=float)
    hit_vals = np.array([float(r["hit_mean_success"]) for r in per_bn], dtype=float)
    gap_vals = np.array([float(r["best_gap_mean"]) for r in per_bn], dtype=float)
    diverged_vals = np.array([float(r["diverged_frac"]) for r in per_bn], dtype=float)

    min_sr = float(np.nanmin(sr_vals)) if sr_vals.size else math.nan
    avg_sr = float(np.nanmean(sr_vals)) if sr_vals.size else math.nan
    finite_hits = hit_vals[np.isfinite(hit_vals)]
    avg_hit = float(np.mean(finite_hits)) if finite_hits.size else math.nan
    worst_hit = float(np.max(finite_hits)) if finite_hits.size else math.nan
    avg_gap = float(np.nanmean(gap_vals)) if gap_vals.size else math.nan
    max_diverged_frac = float(np.nanmax(diverged_vals)) if diverged_vals.size else math.nan

    return {
        "min_sr_hat": min_sr,
        "avg_sr_hat": avg_sr,
        "avg_hit_mean_success": avg_hit,
        "worst_hit_mean_success": worst_hit,
        "avg_best_gap_at_budget": avg_gap,
        "max_diverged_frac": max_diverged_frac,
        "per_benchmark_noise": per_bn,
        "raw_rows": rows,
    }


def generate_candidates(
    *,
    dt_values: list[float],
    lambda_accept_values: list[float],
    lambda_reject_values: list[float],
) -> list[Candidate]:
    out: list[Candidate] = []
    for dt in dt_values:
        for lr in lambda_reject_values:
            for la in lambda_accept_values:
                if not (la > 0.0 and lr > 0.0 and la < lr):
                    continue
                out.append(Candidate(dt=float(dt), lambda_accept=float(la), lambda_reject=float(lr)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["2d", "nd"], default="2d")
    p.add_argument("--dimension", type=int, default=10)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=12345)
    p.add_argument("--n-particles", type=int, default=200)
    p.add_argument("--n-steps", type=int, default=1500)

    p.add_argument("--dt", type=float, nargs="+", default=[0.05, 0.1, 0.1361, 0.2])
    p.add_argument(
        "--lambda-accept",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 2.5],
        help="Direct λ_accept grid (must satisfy 0 < λ_accept < λ_reject).",
    )
    p.add_argument(
        "--lambda-reject",
        type=float,
        nargs="+",
        default=[1.2, 2.0, 3.2, 4.5, 6.0],
        help="Direct λ_reject grid.",
    )
    p.add_argument("--out-dir", type=str, default="results/shared_search")
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    benches = suite_2d() if args.suite == "2d" else suite_nd(int(args.dimension))
    cands = generate_candidates(
        dt_values=[float(x) for x in args.dt],
        lambda_accept_values=[float(x) for x in args.lambda_accept],
        lambda_reject_values=[float(x) for x in args.lambda_reject],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for i, cand in enumerate(cands):
        rows = _evaluate_candidate(
            cand,
            benches,
            repeats=args.repeats,
            seed_base=args.seed_base,
            n_particles=args.n_particles,
            n_steps=args.n_steps,
        )
        summ = summarize_rows(rows)
        per_bn = summ.pop("per_benchmark_noise")
        raw_rows = summ.pop("raw_rows")

        base = {
            "dt": cand.dt,
            "lambda_accept": cand.lambda_accept,
            "lambda_reject": cand.lambda_reject,
            **summ,
        }
        summaries.append(base)

        # Save per-candidate details occasionally (keeps files manageable).
        if i < 3:
            import csv

            raw_path = out_dir / f"raw_{i:04d}.csv"
            with raw_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=sorted({k for r in raw_rows for k in r.keys()}))
                w.writeheader()
                for r in raw_rows:
                    w.writerow(r)

            per_path = out_dir / f"per_bn_{i:04d}.csv"
            with per_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=sorted({k for r in per_bn for k in r.keys()}))
                w.writeheader()
                for r in per_bn:
                    w.writerow(r)

    # Rank: prioritize robustness (min_sr_hat), then speed, then gap.
    summaries.sort(
        key=lambda r: (
            float(r.get("max_diverged_frac", float("inf"))),
            -float(r.get("min_sr_hat", float("nan"))),
            float(r.get("avg_hit_mean_success", float("inf"))),
            float(r.get("avg_best_gap_at_budget", float("inf"))),
        )
    )

    import csv

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in summaries for k in r.keys()}))
        w.writeheader()
        for r in summaries:
            w.writerow(r)

    top_k = summaries[: int(args.top_k)]
    cols = ["dt", "lambda_accept", "lambda_reject", "min_sr_hat", "avg_hit_mean_success"]
    for r in top_k:
        print(" ".join(f"{c}={r.get(c)}" for c in cols))


if __name__ == "__main__":
    main()
