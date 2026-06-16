#!/usr/bin/env python3
"""
Distribution smoke test.

Creates an isolated venv, installs ONLY the freshly built wheel (no editable
install, no repo src/ on the path), and exercises every public API path from a
temp working directory that has no access to the source tree.

Usage:
    python3 scripts/verify_dist.py

Prerequisite:
    python -m build   (or: make build)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"

_UV_SEARCH_PATHS = [
    "uv",
    str(Path.home() / "bin" / "uv"),
    "/a/home/cc/cs/hillelch/bin/uv",
]


def _find_uv() -> str:
    for candidate in _UV_SEARCH_PATHS:
        try:
            subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                check=True,
            )
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError(
        "uv not found. Install it (https://docs.astral.sh/uv/) "
        "or add it to PATH. In CI: pip install uv"
    )


def _run(cmd, *, cwd=None, env=None):
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\nFAILED: {' '.join(str(c) for c in cmd)}", file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result


# ---------------------------------------------------------------------------
# Smoke checks executed inside the clean venv.
# Mirrors every example documented in README.md (labelled by section).
# ---------------------------------------------------------------------------
_SMOKE = """
import sys, os

import tau_community_detection as tau
from tau_community_detection import run_clustering, TauClustering, TauConfig

# Prove we're using the installed wheel, not the repo's src/.
assert "site-packages" in tau.__file__, (
    f"FAIL: tau loaded from {tau.__file__!r} — not from site-packages"
)
print(f"wheel: {tau.__file__}")

# --- 1. Public names exist ---
assert callable(run_clustering)
assert callable(TauClustering)
assert callable(TauConfig)
print("PASS  public names (run_clustering, TauClustering, TauConfig)")

# --- 2. README Quick Start — igraph, bare default (zero-friction snippet) ---
import igraph as ig
g = ig.Graph.Famous("Zachary")
clustering = run_clustering(g)
assert hasattr(clustering, "modularity") and isinstance(clustering.modularity, float)
assert len(clustering) >= 1
print(f"PASS  igraph bare default: {len(clustering)} communities, modularity={clustering.modularity:.4f}")

# --- 3. README NetworkX section — bare default ---
import networkx as nx
gnx = nx.erdos_renyi_graph(n=30, p=0.15, seed=0)
c2 = run_clustering(gnx)
assert len(c2.membership) == gnx.number_of_nodes()
print(f"PASS  networkx bare default: {len(c2)} communities")

# --- 4. Override path (README Quick Start — 'override only the knobs you care about') ---
c3 = run_clustering(
    g,
    resolution=0.8,
    random_seed=42,
    population_size=20,
    max_generations=5,
)
assert hasattr(c3, "membership")
print("PASS  override path (resolution/seed/pop/gen)")

# --- 5. TauConfig field (elite_fraction) via TauClustering ---
#     The original bug lived in the kwargs→TauConfig merge; we verify
#     TauConfig construction with a non-default field works end-to-end.
g_p = ig.Graph.Famous("Petersen")
cfg = TauConfig(
    population_size=10,
    max_generations=3,
    worker_count=1,
    elite_fraction=0.2,
)
c4 = TauClustering(g_p, config=cfg).run()
assert hasattr(c4, "modularity")
print("PASS  TauConfig elite_fraction field via TauClustering")

# --- 6. README Advanced usage — context manager + track_stats ---
config = TauConfig(
    population_size=20,
    max_generations=5,
    resolution=1.0,
    elite_fraction=0.15,
    immigrant_fraction=0.2,
    stopping_generations=3,
    random_seed=42,
    verbose=False,
    worker_count=1,
)
with TauClustering(g, config=config) as t:
    clustering2, stats = t.run(track_stats=True)
assert isinstance(stats, list) and len(stats) >= 1
print(f"PASS  context manager: {len(stats)} generations, modularity={clustering2.modularity:.4f}")

# --- 7. Determinism: same seed → identical membership ---
r1 = run_clustering(g, random_seed=7, population_size=10, max_generations=3)
r2 = run_clustering(g, random_seed=7, population_size=10, max_generations=3)
assert r1.membership == r2.membership, (
    f"same seed produced different results:\\n  r1={r1.membership}\\n  r2={r2.membership}"
)
print("PASS  determinism (same seed → identical membership)")

print()
print("All smoke checks passed.")
"""


def main() -> None:
    wheels = sorted(DIST_DIR.glob("*.whl"))
    if not wheels:
        print(
            "ERROR: No wheel found in dist/. Run 'make build' first.",
            file=sys.stderr,
        )
        sys.exit(1)
    wheel = wheels[-1]
    print(f"Smoke-testing wheel: {wheel.name}\n")

    uv = _find_uv()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        venv_dir = tmp / "venv"
        work_dir = tmp / "work"
        work_dir.mkdir()
        smoke_file = work_dir / "smoke.py"
        smoke_file.write_text(_SMOKE)

        print("Creating isolated venv...")
        _run([uv, "venv", "--python", sys.executable, str(venv_dir)])

        venv_python = venv_dir / "bin" / "python"

        print(f"Installing {wheel.name} (+ deps)...")
        _run([uv, "pip", "install", "--python", str(venv_python), str(wheel)])

        print("Running smoke checks...\n")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        result = subprocess.run(
            [str(venv_python), str(smoke_file)],
            cwd=str(work_dir),
            env=env,
        )
        if result.returncode != 0:
            sys.exit(result.returncode)

        print("\nRunning notebook smoke check (with loky)...\n")
        _run([uv, "pip", "install", "--python", str(venv_python), "nbconvert", "ipykernel", "loky"])
        notebook_src = REPO_ROOT / "scripts" / "smoke_notebook.ipynb"
        notebook_dst = work_dir / "smoke_notebook.ipynb"
        notebook_dst.write_bytes(notebook_src.read_bytes())
        notebook_env = {
            **env,
            "JUPYTER_CONFIG_DIR": str(tmp / "jupyter_config"),
            "JUPYTER_DATA_DIR": str(tmp / "jupyter_data"),
        }
        _run(
            [
                str(venv_python), "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=120",
                str(notebook_dst),
            ],
            cwd=work_dir,
            env=notebook_env,
        )
        print("PASS  notebook (loky active, no warning)")

        # --- Cascade gate: unguarded script + loky shadowed → sentinel once, warns ---
        print("\nRunning cascade gate (loky shadowed)...\n")
        shadow_dir = tmp / "loky_shadow"
        shadow_dir.mkdir()
        (shadow_dir / "loky.py").write_text('raise ImportError("loky shadowed")\n')

        unguarded = work_dir / "cascade_check.py"
        unguarded.write_text(
            "import igraph as ig\n"
            "from tau_community_detection import run_clustering\n"
            "import warnings\n"
            "warnings.simplefilter('always')\n"
            "SENTINEL = 'CASCADE_GATE_RAN'\n"
            "print(SENTINEL, flush=True)\n"
            "result = run_clustering(\n"
            "    ig.Graph.Famous('Petersen'),\n"
            "    population_size=6, max_generations=2, worker_count=4,\n"
            ")\n"
            "assert result is not None\n"
        )
        cascade_env = {
            **env,
            "PYTHONPATH": str(shadow_dir),
        }
        cascade_result = subprocess.run(
            [str(venv_python), str(unguarded)],
            capture_output=True,
            text=True,
            cwd=str(work_dir),
            env=cascade_env,
        )
        if cascade_result.returncode != 0:
            print(f"FAIL  cascade gate exited {cascade_result.returncode}", file=sys.stderr)
            print(cascade_result.stdout, file=sys.stderr)
            print(cascade_result.stderr, file=sys.stderr)
            sys.exit(1)
        sentinel_count = cascade_result.stdout.count("CASCADE_GATE_RAN")
        if sentinel_count != 1:
            print(
                f"FAIL  sentinel appeared {sentinel_count} times (cascade detected)",
                file=sys.stderr,
            )
            print(cascade_result.stdout, file=sys.stderr)
            sys.exit(1)
        combined = cascade_result.stdout + cascade_result.stderr
        if "loky" not in combined.lower() and "sequential" not in combined.lower():
            print("FAIL  no warning about loky/sequential emitted", file=sys.stderr)
            sys.exit(1)
        print("PASS  cascade gate (sentinel once, warning emitted)")

        print("\nAll checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
