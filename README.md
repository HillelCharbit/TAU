# TAU Community Detection

[![PyPI](https://img.shields.io/pypi/v/tau-community-detection.svg)](https://pypi.org/project/tau-community-detection/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Downloads](https://static.pepy.tech/badge/tau-community-detection)](https://pepy.tech/project/tau-community-detection)
[![Build Status](https://img.shields.io/github/actions/workflow/status/HillelCharbit/TAU/python-ci.yml?branch=master)](https://github.com/HillelCharbit/TAU/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

`tau-community-detection` implements TAU, an evolutionary community detection algorithm
that couples genetic search with Leiden refinements. It is designed for scalable graph
clustering with a simple drop-in `run_clustering()` API, sensible defaults, and
multiprocessing support.

---

## Highlights

- **Evolutionary search**: Maintains a population of candidate partitions and applies crossover and mutation tailored for graph clustering.
- **Leiden optimization**: Refines every candidate with Leiden to ensure modularity gains each generation.
- **Multiprocessing aware**: Utilises parallel worker pools for population optimization with automatic fallback to sequential mode.
- **Reproducible runs**: Pass `random_seed` to seed TAU's numpy RNG and igraph's Python-level RNG callbacks for consistent behaviour across runs.
- **Input flexibility**: Accepts `igraph.Graph`, `networkx.Graph`, or a file path. Edge weights are auto-detected.
- **Simple API**: Use `run_clustering(graph)` for zero-friction usage, or drop down to `TauClustering` + `TauConfig` for full control.

---

## Installation

Requires Python 3.10 or newer.

```bash
pip install tau-community-detection
```

To work from a clone:

```bash
git clone https://github.com/HillelCharbit/TAU.git
cd TAU
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

---

## Quick Start

```python
import igraph as ig
from tau_community_detection import run_clustering

g = ig.Graph.Famous("Zachary")

# Zero-friction default usage
clustering = run_clustering(g)
print(f"Communities: {len(clustering)},  Modularity: {clustering.modularity:.4f}")

# Override only the knobs you care about
clustering = run_clustering(
    g,
    resolution_parameter=0.8,
    random_seed=42,
    verbose=True,
    population_size=100,
    max_generations=50,
)
```

`run_clustering()` returns an `igraph.VertexClustering`, so `.membership`, `.modularity`, and all standard igraph attributes are available immediately.

### NetworkX input

```python
import networkx as nx
from tau_community_detection import run_clustering

g = nx.erdos_renyi_graph(n=500, p=0.02, seed=0)
clustering = run_clustering(g)
```

### Advanced usage with `TauClustering`

For full control over the lifecycle — including reusing the worker pool across multiple runs:

```python
from tau_community_detection import TauClustering, TauConfig

config = TauConfig(
    resolution_parameter=1.0,
    elite_fraction=0.15,
    immigrant_fraction=0.2,
    stopping_generations=10,
    random_seed=42,
    verbose=True,
)

with TauClustering(g, population_size=60, max_generations=30, config=config) as tau:
    clustering, stats = tau.run(track_stats=True)

print(f"Ran for {len(stats)} generations")
print(f"Final modularity: {clustering.modularity:.4f}")
```

`track_stats=True` returns a list of per-generation dicts with keys `generation`, `top_fitness`, `average_fitness`, `time_per_generation`, `convergence`, `elite_runtime`, `crossover_runtime`.

---

## Graph Input

Supported sources:

| Type | Notes |
|---|---|
| `igraph.Graph` | Passed directly; weights auto-detected from `"weight"` edge attribute |
| `networkx.Graph` | Converted internally; weights auto-detected |
| `str` (file path) | Edgelist/NCOL (`.graph`, `.edgelist`, `.txt`) or adjacency list (`.adjlist`) |

For large graphs or high worker counts, passing a **file path** is recommended — it avoids serialising the graph object across worker processes.

Edge weights are detected automatically. To override:

```python
from tau_community_detection import TauConfig
config = TauConfig(is_weighted=False)   # force unweighted even if file has weights
```

---

## Configuration Reference

All hyperparameters live on `TauConfig`. Every field is validated on construction — invalid values raise `ValueError` immediately.

| Parameter | Default | Valid range | Description |
|---|---|---|---|
| `population_size` | 60 | > 0 | Number of candidate partitions per generation |
| `max_generations` | 500 | > 0 | Hard cap on evolutionary iterations |
| `worker_count` | `None` | ≥ 1 | Parallel workers (default: CPU count, capped by population size) |
| `elite_fraction` | 0.1 | (0, 1] | Fraction of best partitions preserved each generation |
| `immigrant_fraction` | 0.15 | (0, 1] | Fraction of fresh random partitions injected each generation |
| `selection_power` | 5 | > 0 | Sharpness of fitness-proportional parent selection |
| `elite_similarity_threshold` | 0.9 | [0, 1] | Jaccard threshold below which two elites are considered diverse |
| `stopping_generations` | 10 | > 0 | Generations without improvement before early stopping |
| `stopping_jaccard` | 0.98 | [0, 1] | Similarity threshold that counts as "no improvement" |
| `n_iterations` | 3 | > 0 | Leiden iterations per fitness evaluation |
| `resolution_parameter` | 1.0 | > 0 | Leiden resolution — higher values produce more, smaller communities |
| `sample_fraction_range` | (0.2, 0.9) | 0 < low ≤ high ≤ 1 | Range for random subgraph sampling during population init |
| `is_weighted` | `None` | bool or None | Override weight auto-detection (`None` = auto) |
| `default_edge_weight` | 1.0 | > 0 | Weight assigned to edges when graph is treated as unweighted |
| `weight_attribute` | `"weight"` | str or None | Edge attribute name to read weights from |
| `sim_sample_size` | 20 000 | int or None | Node sample size for Jaccard similarity (None = all nodes) |
| `worker_chunk_size` | `None` | int or None | Tasks per worker per batch (None = auto) |
| `reuse_worker_pool` | `True` | bool | Keep worker pool alive between calls on the same instance |
| `random_seed` | `None` | int or None | Seeds numpy and igraph's Python-level RNG callbacks for consistent runs |
| `verbose` | `False` | bool | Log progress to the standard Python logger |

`run_clustering()` exposes the most common parameters directly. Any `TauConfig` field can also be passed as a keyword argument:

```python
clustering = run_clustering(g, elite_fraction=0.2, stopping_generations=5)
```

---

## Development

```bash
pip install -r requirements-dev.txt
pip install -e .
make lint     # ruff checks
make test     # pytest
make coverage # pytest + coverage report
make build    # build sdist + wheel
```

### Continuous Integration

GitHub Actions runs lint, tests (Python 3.10 and 3.11), and a package build on every push and pull request. Set the `CODECOV_TOKEN` secret to upload coverage reports.

### Publishing

1. Bump `version` in `setup.cfg` and commit.
2. Tag the release: `git tag vX.Y.Z && git push --tags`.
3. Run the **Publish Package** workflow. Use `TEST_PYPI_API_TOKEN` for a dry run on TestPyPI, or `PYPI_API_TOKEN` to publish to PyPI.

---

## Reference & Citation

If you use TAU in your research, please cite:

> **From Leiden to Tel-Aviv University (TAU): exploring clustering solutions via a genetic algorithm**
> Gal Gilad and Roded Sharan. *PNAS Nexus*, Volume 2, Issue 6, June 2023.
> [DOI: 10.1093/pnasnexus/pgad180](https://doi.org/10.1093/pnasnexus/pgad180)

```bibtex
@article{gilad2023tau,
  title={From Leiden to Tel-Aviv University (TAU): exploring clustering solutions via a genetic algorithm},
  author={Gilad, Gal and Sharan, Roded},
  journal={PNAS Nexus},
  volume={2},
  number={6},
  pages={pgad180},
  year={2023},
  publisher={Oxford University Press}
}
```

---

## License

[MIT License](LICENSE) © 2023 Hillel Charbit
