"""Utilities for working with in-memory graphs for TAU."""
from __future__ import annotations

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
    graph_source: ig.Graph | nx.Graph,
    *,
    weight_attribute: Optional[str] = "weight",
    default_weight: float = 1.0,
) -> ig.Graph:
    """Normalise an in-memory graph into an ``igraph.Graph``.

    Parameters
    ----------
    graph_source:
        Either an ``igraph.Graph`` or a ``networkx.Graph`` that is already loaded in memory.
    """
    if isinstance(graph_source, ig.Graph):
        ig_graph = graph_source.copy()
        if weight_attribute is not None and weight_attribute not in ig_graph.es.attributes():
            ig_graph.es[weight_attribute] = [float(default_weight)] * ig_graph.ecount()
        if (
            weight_attribute is not None
            and weight_attribute != "weight"
            and "weight" not in ig_graph.es.attributes()
        ):
            ig_graph.es["weight"] = list(ig_graph.es[weight_attribute])
        return ig_graph
    if isinstance(graph_source, nx.Graph):
        default_weight = float(default_weight)
        return networkx_to_igraph(
            graph_source,
            weight_attribute=weight_attribute,
            default_weight=default_weight,
        )
    raise TypeError(
        "load_graph expects an igraph.Graph or networkx.Graph instance; file paths are not supported."
    )


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
