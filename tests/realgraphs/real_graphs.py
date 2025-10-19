from __future__ import annotations

import gzip
from pathlib import Path
import sys
from typing import Iterable

import igraph as ig
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tau_community_detection.algorithm import TauClustering, TauConfig


def _iter_snap_edges(handle: Iterable[str]) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for line in handle:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            source = int(parts[0])
            target = int(parts[1])
        except ValueError:
            continue
        if source == target:
            continue
        a, b = (source, target) if source < target else (target, source)
        edges.append((a, b))
    return edges


def _read_snap_file() -> list[tuple[int, int]] | None:
    directory = Path(__file__).parent
    txt_path = directory / "as-22july06.txt"
    gz_path = directory / "as-22july06.txt.gz"

    if txt_path.exists():
        with txt_path.open("r", encoding="ascii") as handle:
            return _iter_snap_edges(handle)
    if gz_path.exists():
        with gzip.open(gz_path, "rt", encoding="ascii") as handle:
            return _iter_snap_edges(handle)
    return None


def _read_legacy_fixture() -> list[tuple[int, int]]:
    fixture_path = Path(__file__).with_name("as-22july06.graph")
    if not fixture_path.exists():
        raise FileNotFoundError(
            "Place as-22july06.txt(.gz) from SNAP or keep the legacy as-22july06.graph fixture next to this script."
        )

    edges: set[tuple[int, int]] = set()
    with fixture_path.open("r", encoding="ascii") as fixture:
        for line_index, line in enumerate(fixture):
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                # Some trailing identifiers have no adjacency entries
                continue
            if line_index == 0 and len(parts) >= 3:
                # First line contains metadata counts (nodes, edges, directed flag)
                continue
            try:
                source = int(parts[0])
            except ValueError:
                continue
            for value in parts[1:]:
                try:
                    target = int(value)
                except ValueError:
                    continue
                if source == target:
                    continue
                edge = (source, target) if source < target else (target, source)
                edges.add(edge)

    if not edges:
        raise ValueError("Legacy fixture graph has no edges; cannot run integration test.")
    return list(edges)


def _load_fixture_graph() -> ig.Graph:
    edges = _read_snap_file()
    if edges is None:
        edges = _read_legacy_fixture()

    if not edges:
        raise ValueError("Graph file contains no edges; cannot run integration test.")

    nodes = {node for edge in edges for node in edge}
    node_map = {node: idx for idx, node in enumerate(sorted(nodes))}
    relabeled_edges = [(node_map[u], node_map[v]) for u, v in edges]

    graph = ig.Graph(edges=relabeled_edges, directed=False)
    graph.simplify(multiple=True, loops=True)
    return graph


def test_real_graph():
    graph = _load_fixture_graph()

    config = TauConfig(
        population_size=10,
        max_generations=10,
        random_seed=42,
        stopping_generations=10,
    )

    tau = TauClustering(
        graph,
        population_size=config.population_size,
        max_generations=config.max_generations,
        config=config,
    )

    membership, modularity_history = tau.run()

    if membership.shape != (graph.vcount(),):
        raise RuntimeError("Membership vector does not match graph order.")
    if not modularity_history:
        raise RuntimeError("TAU clustering did not report any modularity values.")
    best_modularity = modularity_history[-1]
    if not np.isfinite(best_modularity):
        raise RuntimeError("Best modularity is not finite.")
    if best_modularity != max(modularity_history):
        raise RuntimeError("Modularity history does not end with the best observed value.")

    print(f"nodes={graph.vcount()} edges={graph.ecount()} best_modularity={best_modularity:.6f}")


if __name__ == "__main__":
    test_real_graph()
