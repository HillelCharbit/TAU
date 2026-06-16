"""Regression tests for parallel backend selection.

Two properties pinned:
1. run_clustering() bare defaults never raises (guards against worker_count kwarg bugs).
2. Unguarded script + no loky + spawn → warns, runs sequentially, sentinel printed once.
   Each assertion is independent so a regression only fails its own line.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import igraph as ig
import pytest

from tau_community_detection import run_clustering
from tau_community_detection.algorithm import _choose_backend, _main_has_guard


# ---------------------------------------------------------------------------
# 1. Bare-default regression
# ---------------------------------------------------------------------------

def test_bare_defaults_does_not_raise():
    g = ig.Graph.Famous("Petersen")
    result = run_clustering(g)
    assert hasattr(result, "modularity")
    assert len(result.membership) == g.vcount()


# ---------------------------------------------------------------------------
# 2. _choose_backend unit tests
# ---------------------------------------------------------------------------

def test_choose_backend_sequential_when_one_worker():
    backend, n = _choose_backend(1)
    assert backend == "sequential"
    assert n == 1


def test_choose_backend_sequential_when_zero_workers():
    backend, n = _choose_backend(0)
    assert backend == "sequential"
    assert n == 1


def test_main_has_guard_returns_true_for_no_file():
    """Notebooks / REPL have no __file__ → treated as guarded."""
    import types
    fake_main = types.ModuleType("__main__")
    # no __file__ attribute
    import sys
    original = sys.modules.get("__main__")
    sys.modules["__main__"] = fake_main
    try:
        assert _main_has_guard() is True
    finally:
        if original is None:
            del sys.modules["__main__"]
        else:
            sys.modules["__main__"] = original


# ---------------------------------------------------------------------------
# 3. Cascade subprocess test
# ---------------------------------------------------------------------------

_UNGUARDED_SCRIPT = textwrap.dedent("""\
    # UNGUARDED: no if __name__ == '__main__': guard on purpose.
    import igraph as ig
    from tau_community_detection import run_clustering
    import warnings
    warnings.simplefilter("always")

    SENTINEL = "TAU_RAN_ONCE"
    print(SENTINEL, flush=True)

    result = run_clustering(
        ig.Graph.Famous("Petersen"),
        population_size=6,
        max_generations=2,
        worker_count=4,
    )
    assert result is not None
""")

_LOKY_SHADOW = textwrap.dedent("""\
    # Shadow loky so it appears uninstalled.
    raise ImportError("loky shadowed for test")
""")


@pytest.fixture()
def cascade_env(tmp_path: Path):
    """Write the unguarded script and a loky shadow module to tmp_path."""
    script = tmp_path / "unguarded.py"
    script.write_text(_UNGUARDED_SCRIPT)

    loky_shadow = tmp_path / "loky.py"
    loky_shadow.write_text(_LOKY_SHADOW)

    return tmp_path


def _run_script(script: Path) -> subprocess.CompletedProcess:
    import os
    # src/ must come first so the subprocess uses the current source tree, not
    # any stale installed wheel.  The loky shadow (loky.py in script.parent)
    # comes second — it is still found before site-packages, making loky appear
    # absent to the subprocess.
    src_dir = str(Path(__file__).parent.parent / "src")
    pythonpath = src_dir + os.pathsep + str(script.parent)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
    )


def test_cascade_no_loky_spawn_exits_zero(cascade_env: Path):
    """Unguarded script + shadowed loky must exit 0, not crash."""
    result = _run_script(cascade_env / "unguarded.py")
    assert result.returncode == 0, (
        f"Script exited {result.returncode}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_cascade_sentinel_printed_exactly_once(cascade_env: Path):
    """Sentinel must appear exactly once — cascade would print it N times."""
    result = _run_script(cascade_env / "unguarded.py")
    count = result.stdout.count("TAU_RAN_ONCE")
    assert count == 1, (
        f"Sentinel appeared {count} times (expected 1 — cascade detected).\n"
        f"stdout:\n{result.stdout}"
    )


def test_cascade_no_loky_emits_warning(cascade_env: Path):
    """Unguarded + no loky + spawn must emit a RuntimeWarning about sequential fallback."""
    result = _run_script(cascade_env / "unguarded.py")
    combined = result.stdout + result.stderr
    assert "loky" in combined.lower() or "sequential" in combined.lower(), (
        f"Expected a warning mentioning loky or sequential fallback.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
