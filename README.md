# Optimizer comparison (ALO vs CBO)

This repository compares a **particle-based adaptive leader optimizer (ALO)** (included here under **`alo/`**) against **consensus-based optimization (CBO)** from **CBXpy** (cloned into **`external/CBXpy`**). Experiments live under `experiments/`; shared types and runners are in `common/`.

## External dependency (clone first)

From the **repository root**, clone CBXpy into the expected path:

```bash
git clone https://github.com/PdIPS/CBXpy.git external/CBXpy
```

## Prerequisites

- **Python** 3.10+ recommended (code uses modern typing syntax).
- **Packages:** **NumPy**, **SciPy**, and **Matplotlib** (pinned loosely in **`requirements.txt`**). Install:

  ```bash
  pip install -r requirements.txt
  ```

Nothing needs to be `pip install`’d from `external/CBXpy`; the runners prepend that directory to `sys.path` when CBO runs.

## Repository layout

| Path | Role |
|------|------|
| `main.py` | Adds the repo root to `sys.path` so imports work consistently (same pattern as scripts under `experiments/`). |
| `alo/` | ALO implementation: `algorithm.py` (simulator) and `test_fun.py` (benchmarks). |
| `common/` | `Benchmark` types, diagnostics, **`custom_runner`** (ALO → `simulate_particles`), **`cbx_runner`** (CBO via CBXpy), **`consensus_metrics`** (softmax consensus statistics). |
| `experiments/` | CLI scripts for suites, sweeps, tables, animations. **`benchmarks.py`** defines `suite_2d()` / `suite_nd()` using objectives from `test_fun.py`. |
| `external/CBXpy/` | **Git clone** ([PdIPS/CBXpy](https://github.com/PdIPS/CBXpy)); CBO dynamics. Ignored by this repo’s git. |

Benchmark functions (Beale, Himmelblau, Rastrigin, Rosenbrock in 2D, etc.), bounds, and `f★` wiring are centralized in **`experiments/benchmarks.py`**.

## Outputs

Scripts write under **`results/`** (created automatically; listed in **`.gitignore`** so regenerated outputs stay local).

---

## Running experiments

Run these from the **repository root** so path bootstrapping matches the scripts.

### 1. Default ALO vs CBO comparison CSV

Runs each 2D benchmark for both noise modes, both methods, shared random initialization per `(benchmark, repeat)`.

```bash
python experiments/run_compare.py --help
python experiments/run_compare.py --suite 2d --repeats 5 --out results/compare_2d.csv
```

High-dimensional suite:

```bash
python experiments/run_compare.py --suite nd --dimension 10 --repeats 3 --out results/compare_nd_d10.csv
```

### 2. Consensus-metrics table (four algorithms × four 2D problems)

Produces a Markdown summary and per-run CSV: SR, median **consensus first-hit step**, median **best consensus gap** over the horizon (`common/consensus_metrics.py`). Uses softmax consensus with **α** matching CBO defaults.

```bash
python experiments/run_continuous_benchmark_table.py --help
python experiments/run_continuous_benchmark_table.py --repeats 100 --out-md results/continuous_benchmark_table.md --out-csv results/continuous_benchmark_runs.csv
```

Parallelism defaults to roughly `CPU count − 1` (`--workers`).

### 3. Particle-count sweep (ALO only)

Sweep **N**, isotropic vs anisotropic ALO noise, runtime and mean final swarm-best objective.

```bash
python experiments/beale_N_particle_tradeoff.py --help
python experiments/beale_N_particle_tradeoff.py --benchmark rastrigin --repeats 20 --out-csv results/rastrigin_alo_N_sweep.csv --out-plot results/rastrigin_alo_N_tradeoff.png
```

Default `--benchmark` is **rastrigin**; use `--benchmark beale` (etc.) for other names in `suite_2d()`.

### 4. Other scripts (see `--help`)

- **`experiments/simple_lambda_grid_scan.py`** — λ rejection grid on chosen benchmarks.
- **`experiments/search_custom_shared.py`** — hyperparameter search over ALO ratios (multi-benchmark CSVs).
- **`experiments/ablation_equal_lambda_N50_2d.py`** — controlled ablations.
- **`experiments/animate_2d_trajectories.py`** — 2D trajectory animations (GIFs under `results/animations/` by default).

---

## Noise models (important caveat)

| Method | `"isotropic"` | `"anisotropic"` |
|--------|----------------|-----------------|
| **ALO** | Scalar spread from swarm covariance × `√dt` × `𝒩(0,1)` per particle coordinate (see `algorithm.py`). | Per-step covariance diffusion from particle spread (non-diagonal coupling). |
| **CBO (this repo)** | CBXpy string **`"isotropic"`** (noise aligned with swarm scale). | **Custom callable:** component-wise \(\sqrt{\mathrm{dt}}\,\mathcal{N}(0,I)\), *not* CBXpy’s string `"anisotropic"` (which scales noise by drift). |

So “anisotropic CBO” in this codebase means **component-wise Brownian increments**, chosen to mirror ALO’s axis-wise stochasticity more closely than CBXpy’s drift-scaled variant.

---

## Reproducing a minimal import check

```bash
python -c "from common.custom_runner import run_custom; from common.cbx_runner import run_cbx_cbo; print('ok')"
```

If that fails, ensure you are at the repo root and that `external/CBXpy` is present.
