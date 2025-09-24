"""Utilities for loading graph data into igraph."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import igraph as ig
import networkx as nx


def load_graph(
    path: str | Path,
    *,
    weight_attribute: Optional[str] = "weight",
    default_weight: float = 1.0,
) -> ig.Graph:
    """Load a graph file (weighted edgelist or adjacency list) into an igraph.Graph.

    Parameters
    ----------
    path: str | Path
        Path to a NetworkX-compatible adjacency list or weighted edgelist file.
    """
    adj_path = Path(path)
    if not adj_path.exists():
        raise FileNotFoundError(f"Graph file not found: {adj_path}")

    if adj_path.suffix == ".igpkl":
        return ig.Graph.Read_Pickle(str(adj_path))

    default_weight = float(default_weight)

    nx_graph: Optional[nx.Graph] = None
    try:
        nx_graph = nx.read_weighted_edgelist(adj_path)
    except (ValueError, nx.NetworkXError, IndexError):
        nx_graph = None

    if nx_graph is None:
        nx_graph = nx.read_adjlist(adj_path)
        is_weighted_source = False
    else:
        is_weighted_source = True

    use_weights = weight_attribute is not None
    if use_weights:
        for u, v, data in nx_graph.edges(data=True):
            base = data.get("weight", default_weight)
            if not is_weighted_source and "weight" not in data:
                data["weight"] = float(base)
            weight_val = float(data.get(weight_attribute, base))
            data[weight_attribute] = weight_val

    return networkx_to_igraph(nx_graph, weight_attribute, default_weight)


def _edges_to_lists(
    graph: nx.Graph, weight_attribute: Optional[str], default_weight: float
) -> tuple[list[int], list[int], Optional[list[float]]]:
    """Convert an edge list to parallel source/target lists (optionally weighted)."""
    sources: list[int] = []
    targets: list[int] = []
    weights: Optional[list[float]] = [] if weight_attribute is not None else None
    default_weight = float(default_weight)
    for source, target, data in graph.edges(data=True):
        sources.append(int(source))
        targets.append(int(target))
        if weights is not None:
            weight_val = float(data.get(weight_attribute, default_weight))
            weights.append(weight_val)
    return sources, targets, weights


def networkx_to_igraph(
    graph: nx.Graph,
    weight_attribute: Optional[str] = "weight",
    default_weight: float = 1.0,
) -> ig.Graph:
    """Convert a NetworkX graph into an igraph.Graph with optional weights."""
    default_weight = float(default_weight)
    use_weights = weight_attribute is not None

    working = graph.copy()
    if use_weights:
        for _, _, data in working.edges(data=True):
            weight_val = float(data.get(weight_attribute, data.get("weight", default_weight)))
            data[weight_attribute] = weight_val
            data["weight"] = weight_val
    else:
        for _, _, data in working.edges(data=True):
            if "weight" in data:
                del data["weight"]

    mapping = {node: idx for idx, node in enumerate(working.nodes())}
    working = nx.relabel_nodes(working, mapping)

    sources, targets, weights = _edges_to_lists(working, weight_attribute if use_weights else None, default_weight)
    ig_graph = ig.Graph(len(working), list(zip(sources, targets)))
    if use_weights and weights is not None:
        ig_graph.es[weight_attribute] = weights
    return ig_graph
