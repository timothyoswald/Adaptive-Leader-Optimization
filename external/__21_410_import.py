from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "21-410-project"

if not SRC_DIR.exists():
    raise RuntimeError(f"Expected 21-410-project at {SRC_DIR}, but it does not exist.")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Re-export the pieces we need from the custom optimizer implementation.
from algorithm import ParticleConfig, simulate_particles  # noqa: E402

__all__ = ["ParticleConfig", "simulate_particles"]

