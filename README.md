# TAU Community Detection

[![PyPI](https://img.shields.io/pypi/v/tau-community-detection.svg)](https://pypi.org/project/tau-community-detection/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Downloads](https://static.pepy.tech/badge/tau-community-detection)](https://pepy.tech/project/tau-community-detection)
[![Build Status](https://img.shields.io/github/actions/workflow/status/HillelCharbit/TAU/python-ci.yml?branch=master)](https://github.com/HillelCharbit/TAU/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


`tau-community-detection` implements TAU, an evolutionary community detection algorithm
that couples genetic search with Leiden refinements. It is designed for scalable graph
clustering with a simple drop-in `run_clustering()` API, sensible defaults, and
multiprocessing support.

---

## Highlights

- **Evolutionary search**: Maintains a population of candidate partitions and applies
  crossover/mutation tailored for graph clustering.
- **Leiden optimization**: Refines every candidate with Leiden to ensure modularity gains.
- **Multiprocessing aware**: Utilises worker pools for population optimization.
- **Deterministic options**: Accepts a user-specified random seed for reproducibility.
- **Simple API**: Use `tau.run_clustering(graph)` for the default workflow, or drop
  down to `TauClustering` and `TauConfig` when you need advanced control.

---

## Installation

The project targets Python 3.10 or newer.

```bash
pip install tau-community-detection
```

To work from a clone, install the package in editable mode inside a virtual environment:

```bash
git clone https://github.com/HillelCharbit/community_TAU.git
cd community_TAU
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start (Python API)

```python
import networkx as nx
import tau_community_detection as tau

g = nx.erdos_renyi_graph(n=1000, p=0.01, seed=42)

# Zero-friction default usage
clustering = tau.run_clustering(g)
print(f"Modularity: {clustering.modularity:.4f}")
print(f"Communities: {len(clustering)}")

# Only override the knobs you care about
clustering = tau.run_clustering(
  g,
  resolution_parameter=0.8,
  random_seed=42,
  verbose=True,
  population_size=100,
  max_generations=50,
)
```

`run_clustering()` returns an `igraph.VertexClustering` object, so the usual
`modularity` and `membership` attributes are available immediately.

For advanced tuning, pass additional `TauConfig` fields directly as keyword
arguments without having to instantiate `TauConfig` yourself:

```python
clustering = tau.run_clustering(
  g,
  stopping_generations=5,
  stopping_jaccard=0.95,
  elite_fraction=0.2,
)
```

---

## Graph input
 
To optimize for very large graphs or when using many worker processes, it is recommended to pass a file path (e.g., to an `.graph`, `.ncol`, or `.edgelist` file) directly to `run_clustering()` or `TauClustering` rather than a pre-loaded graph object. This allows efficient memory sharing.
 
Supported input:
- File path to a graph in common NetworkX or igraph format (auto-detects weighting and structure).
- Already-loaded `networkx.Graph` or `igraph.Graph` objects.

By default, the loader auto-detects whether the graph is weighted based on the file or graph structure. You can override this by setting `TauConfig(is_weighted=True/False)` when constructing `TauClustering`, or by passing the appropriate weight settings into `run_clustering()`.

See the **Quick Start** section above for usage examples.

---

## Configuration

All algorithm hyper-parameters live on the `TauConfig` dataclass. The high-level
`run_clustering()` wrapper accepts the most common ones directly, while `TauConfig`
remains available for advanced workflows. Key fields include:

- `worker_count`: number of parallel processes (defaults to CPU count, capped by population size).
- `population_size`: number of partitions maintained per generation (default: 60 in `run_clustering()`).
- `max_generations`: upper bound on evolutionary iterations (default: 20 in `run_clustering()`).
- `verbose`: set to `True` for progress logging (default: False).
- `stopping_generations` / `stopping_jaccard`: convergence checks based on membership
  stability.
- `random_seed`: makes runs reproducible across processes.

See `src/tau_community_detection/config.py` for the complete list.

---

## Development

```bash
pip install -r requirements-dev.txt
make lint
make test
```

To build local distributions:

```bash
make build
```

### Continuous Integration

- GitHub Actions run lint, tests, and package builds on pushes and pull requests.
- Set the `CODECOV_TOKEN` secret to upload coverage reports.

### Publishing

1. Bump the version in `setup.cfg`/`pyproject.toml` and commit.
2. Tag the release with `git tag vX.Y.Z && git push --tags`.
3. Run the **Publish Package** workflow (defaults to TestPyPI). For PyPI, supply the `pypi`
   input and ensure `PYPI_API_TOKEN` is set. Use `TEST_PYPI_API_TOKEN` for dry runs.

---
## Reference & Citation

If you use TAU in your research, please cite the original algorithm paper:

> **From Leiden to Tel-Aviv University (TAU): exploring clustering solutions via a genetic algorithm**
> Gal Gilad and Roded Sharan. *PNAS Nexus*, Volume 2, Issue 6, June 2023.
> [DOI: 10.1093/pnasnexus/pgad180](https://doi.org/10.1093/pnasnexus/pgad180)

**BibTeX:**
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
## License & Versioning

**Current Version:** 1.3.2
**License:** This project is licensed under the [MIT License](LICENSE).

See the [Changelog](CHANGELOG.md) for a detailed history of changes and updates.