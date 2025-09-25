"""Utilities for loading graph data into igraph."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import igraph as ig
import networkx as nx


def _resolve_weight(
    data: dict,
    weight_attribute: Optional[str],
    default_weight: float,
) -> float:
    if weight_attribute is None:
        return float(default_weight)
    if weight_attribute in data:
        return float(data[weight_attribute])
    if "weight" in data:
        return float(data["weight"])
    return float(default_weight)


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


def networkx_to_igraph(
    graph: nx.Graph,
    weight_attribute: Optional[str] = "weight",
    default_weight: float = 1.0,
) -> ig.Graph:
    """Convert a NetworkX graph into an igraph.Graph with optional weights."""
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
    edge_count = graph.number_of_edges()

    sources: list[int] = [0] * edge_count
    targets: list[int] = [0] * edge_count
    weights: Optional[list[float]]
    if weight_attribute is not None:
        weights = [0.0] * edge_count
    else:
        weights = None

    default_weight = float(default_weight)
    for idx, (source, target, data) in enumerate(graph.edges(data=True)):
        sources[idx] = node_mapping[source]
        targets[idx] = node_mapping[target]
        if weights is not None:
            weights[idx] = _resolve_weight(data, weight_attribute, default_weight)

    ig_graph = ig.Graph(n=len(node_mapping), directed=graph.is_directed())
    ig_graph.add_edges(zip(sources, targets))

    if weights is not None:
        ig_graph.es["weight"] = weights
        if weight_attribute != "weight":
            ig_graph.es[weight_attribute] = weights

    return ig_graph
