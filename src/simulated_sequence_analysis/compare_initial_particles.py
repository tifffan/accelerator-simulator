#!/usr/bin/env python3
"""compare_initial_vs_writebeam.py — up to five write_beam snapshots
===================================================================
Compare *initial_particles* with the first **up to five** `write_beam_N`
ParticleGroups in a single Impact archive. Optionally save a PNG that shows the
phase‑space projections side‑by‑side.

Figure layout
-------------
``Initial`` | ``Write‑1`` | ``Write‑2`` | ``Write‑3`` | ``Write‑4`` | ``Write‑5``
:----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:
(x,px)      | (x,px)      | …           |             |             |            
(y,py)      | (y,py)      | …           |             |             |            
(z,pz)      | (z,pz)      | …           |             |             |            

If fewer than five `write_beam_*` groups exist the figure will include as many
columns as available.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from impact import Impact
from pmd_beamphysics import ParticleGroup

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

VARIABLES = ["x", "y", "z", "px", "py", "pz"]
PAIR_LABELS = [("x", "px"), ("y", "py"), ("z", "pz")]


def _save_pg(img_pg: ParticleGroup, xvar: str, pvar: str) -> str:
    """Render pg.plot→PNG, return path."""
    fig = img_pg.plot(xvar, pvar, return_figure=True)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _norm_emit(pg: ParticleGroup, coord: str) -> float:
    mom = {"x": "px", "y": "py", "z": "pz"}[coord]
    a, pa = pg[coord], pg[mom]
    return float(np.sqrt(np.mean(a**2) * np.mean(pa**2) - np.mean(a * pa) ** 2))


def _align(vals: np.ndarray, ids: np.ndarray, ref: np.ndarray) -> np.ndarray:
    idx = {pid: i for i, pid in enumerate(ids)}
    sel = np.fromiter((idx[p] for p in ref), dtype=np.int64)
    return vals[sel]


def mse_between(pg_a: ParticleGroup, pg_b: ParticleGroup) -> float:
    ref = np.intersect1d(pg_a.id, pg_b.id, assume_unique=True)
    if ref.size == 0:
        return np.nan
    arr_a = np.vstack([_align(getattr(pg_a, v), pg_a.id, ref) for v in VARIABLES]).T
    arr_b = np.vstack([_align(getattr(pg_b, v), pg_b.id, ref) for v in VARIABLES]).T
    return float(np.mean((arr_a - arr_b) ** 2))


def plot_columns(initial: ParticleGroup, writes: Sequence[ParticleGroup], out: Path) -> None:
    n_cols = 1 + len(writes)
    fig, axes = plt.subplots(nrows=3, ncols=n_cols, figsize=(6 * n_cols, 18))
    all_pgs = [initial] + list(writes)
    labels = ["Initial"] + [f"Write‑{i+1}" for i in range(len(writes))]

    for r, (xv, pv) in enumerate(PAIR_LABELS):
        for c, (pg, lab) in enumerate(zip(all_pgs, labels)):
            ax = axes[r, c]
            img_path = _save_pg(pg, xv, pv)
            ax.imshow(mpimg.imread(img_path))
            ax.set_title(f"{lab} {xv}-{pv}")
            ax.axis("off")
            os.remove(img_path)

    # Caption: metrics vs each write
    lines = []
    for i, pg in enumerate(writes):
        mse = mse_between(initial, pg)
        rel = [abs(_norm_emit(initial, c) - _norm_emit(pg, c)) / (_norm_emit(pg, c) + 1e-30) for c in ("x", "y", "z")]
        lines.append(f"Initial ↔ Write‑{i+1}: MSE={mse:.3e}, rel‑err emitt (x,y,z)={rel[0]:.3e},{rel[1]:.3e},{rel[2]:.3e}")
    fig.text(0.5, 0.04, "\n".join(lines), ha="center", va="center", fontsize=11,
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out)
    plt.close(fig)

# -----------------------------------------------------------------------------
# CLI logic
# -----------------------------------------------------------------------------

def write_beam_keys(keys: List[str]) -> List[str]:
    return sorted([k for k in keys if k.startswith("write_beam_")], key=lambda k: int(re.search(r"\d+", k).group()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare initial vs up to five write_beam groups; optionally plot.")
    ap.add_argument("--h5_file")
    ap.add_argument("--plot_dir", help="Directory to save PNG figure")
    args = ap.parse_args()

    if not os.path.isfile(args.h5_file):
        sys.exit("File not found")

    try:
        I = Impact.from_archive(args.h5_file)
    except Exception as e:
        sys.exit(f"Cannot open file: {e}")

    if "initial_particles" not in I.particles:
        sys.exit("Archive missing initial_particles group")

    w_keys = write_beam_keys(list(I.particles))[:5]  # take first five
    if not w_keys:
        sys.exit("No write_beam_* groups found")

    initial = I.particles["initial_particles"].copy()
    writes = [I.particles[k] for k in w_keys]

    # Drift initial if needed
    if "z" not in initial.columns and "z" in writes[0].columns:
        initial.drift_to_z(float(writes[0].avg("z")))
    else:
        initial.drift_to_t(float(initial.avg("t")))

    print("Comparison columns: Initial +", ", ".join(w_keys))

    if args.plot_dir:
        Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(args.plot_dir) / f"{Path(args.h5_file).stem}_initial_vs_writebeams.png"
        plot_columns(initial, writes, out_path)
        print("Plot saved →", out_path)


if __name__ == "__main__":
    main()
