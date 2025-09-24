"""Utilities for loading graph data into igraph."""
from __future__ import annotations

from pathlib import Path
import igraph as ig
import networkx as nx


def load_graph(path: str | Path) -> ig.Graph:
    """Load an adjacency list graph file into an igraph.Graph.

    Parameters
    ----------
    path: str | Path
        Path to a NetworkX-compatible adjacency list file.
    """
    adj_path = Path(path)
    if not adj_path.exists():
        raise FileNotFoundError(f"Graph file not found: {adj_path}")

    nx_graph = nx.read_adjlist(adj_path)
    # Remap nodes to contiguous integer ids for igraph compatibility
    mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}
    nx_graph = nx.relabel_nodes(nx_graph, mapping)

    sources, targets = _edges_to_lists(nx_graph)
    graph = ig.Graph(len(nx_graph), list(zip(sources, targets)))
    return graph


def _edges_to_lists(graph: nx.Graph) -> tuple[list[int], list[int]]:
    """Convert an edge list to parallel source/target lists."""
    sources: list[int] = []
    targets: list[int] = []
    for source, target in graph.edges():
        sources.append(int(source))
        targets.append(int(target))
    return sources, targets
