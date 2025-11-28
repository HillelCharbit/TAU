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


def _detect_file_format(path: str) -> tuple[str, bool]:
    """Peek at a file on disk and infer adjacency/edge list format."""
    total_lines = 0
    numeric_three_col = True
    all_three_col = True
    all_two_col = True
    with open(path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            columns = line.split()
            count = len(columns)
            total_lines += 1
            if count >= 4:
                return "adjlist", False
            if count == 3:
                all_two_col = False
                try:
                    float(columns[2])
                except ValueError:
                    numeric_three_col = False
            else:
                all_three_col = False
                if count != 2:
                    all_two_col = False
    if total_lines and all_three_col and numeric_three_col:
        return "edgelist", True
    if total_lines and all_two_col:
        return "edgelist", False
    return "adjlist", False


def _nx_has_weights(graph: nx.Graph, weight_attribute: Optional[str]) -> bool:
    for _, _, data in graph.edges(data=True):
        if (weight_attribute and weight_attribute in data) or ("weight" in data):
            return True
    return False


def _ig_has_weights(graph: ig.Graph, weight_attribute: Optional[str]) -> bool:
    attrs = set(graph.es.attributes())
    if weight_attribute and weight_attribute in attrs:
        return True
    return "weight" in attrs


def _finalize_weights(
    graph: ig.Graph,
    weight_attribute: Optional[str],
    default_weight: float,
    is_weighted: bool,
) -> ig.Graph:
    edge_count = graph.ecount()
    if edge_count == 0:
        return graph
    default_weight = float(default_weight)

    def _resolve_edge_values(source_attr: Optional[str]) -> list[float]:
        if source_attr:
            return [float(val) for val in graph.es[source_attr]]
        return [default_weight] * edge_count

    targets: list[str] = ["weight"]
    if weight_attribute and weight_attribute != "weight":
        targets.insert(0, weight_attribute)
    if is_weighted:
        source_attr: Optional[str]
        if weight_attribute and weight_attribute in graph.es.attributes():
            source_attr = weight_attribute
        elif "weight" in graph.es.attributes():
            source_attr = "weight"
        else:
            source_attr = None
        values = _resolve_edge_values(source_attr)
    else:
        values = [default_weight] * edge_count
    for attribute in targets:
        graph.es[attribute] = list(values)
    return graph


def load_graph(
    graph_source: ig.Graph | nx.Graph | str,
    *,
    weight_attribute: Optional[str] = "weight",
    default_weight: float = 1.0,
    is_weighted: Optional[bool] = None,
    return_is_weighted: bool = False,
) -> ig.Graph | tuple[ig.Graph, bool]:
    """Normalise an in-memory graph into an ``igraph.Graph``.

    Parameters
    ----------
    graph_source:
        Either an ``igraph.Graph`` or a ``networkx.Graph`` that is already loaded in memory,
        or a file path pointing at an edgelist/adjlist file on disk.
    is_weighted:
        When ``None`` the loader inspects file-based graphs to auto-detect whether weights
        are present. A boolean forces the desired policy.
    return_is_weighted:
        When ``True`` return a ``(graph, resolved_bool)`` tuple describing the effective
        weight handling behaviour.
    """
    default_weight = float(default_weight)

    def _final_return(graph: ig.Graph, detected: bool) -> ig.Graph | tuple[ig.Graph, bool]:
        resolved = detected if is_weighted is None else bool(is_weighted)
        graph = _finalize_weights(graph, weight_attribute, default_weight, resolved)
        return (graph, resolved) if return_is_weighted else graph

    if isinstance(graph_source, str):
        fmt, detected = _detect_file_format(graph_source)
        if fmt == "adjlist":
            nx_graph = nx.read_adjlist(graph_source)
            detected = False
        else:
            reader = nx.read_weighted_edgelist if detected else nx.read_edgelist
            nx_graph = reader(graph_source, nodetype=int)
        ig_graph = networkx_to_igraph(
            nx_graph,
            weight_attribute=weight_attribute,
            default_weight=default_weight,
        )
        return _final_return(ig_graph, detected)
    if isinstance(graph_source, ig.Graph):
        ig_graph = graph_source.copy()
        return _final_return(ig_graph, _ig_has_weights(ig_graph, weight_attribute))
    if isinstance(graph_source, nx.Graph):
        nx_graph = graph_source.copy()
        ig_graph = networkx_to_igraph(
            nx_graph,
            weight_attribute=weight_attribute,
            default_weight=default_weight,
        )
        return _final_return(ig_graph, _nx_has_weights(nx_graph, weight_attribute))
    raise TypeError(
        "load_graph expects an igraph.Graph or networkx.Graph instance or a file path string."
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
