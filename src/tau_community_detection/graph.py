""" Graph loading for TAU."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import igraph as ig
import networkx as nx

_TEMP_GRAPH_PATH: Optional[Path] = None

def _detect_weighted(path: str) -> bool:
    """Check if edgelist file has weight column (3 numeric columns)."""
    with open(path, "r") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) >= 3:
                try:
                    float(cols[2])
                    return True
                except ValueError:
                    pass
            return False
    return False

def _graph_to_temp_file(graph: ig.Graph, weighted: bool) -> str:
    """Write igraph to temp NCOL file, return path."""
    global _TEMP_GRAPH_PATH
    fd = tempfile.NamedTemporaryFile(mode="w", suffix=".ncol", delete=False)
    _TEMP_GRAPH_PATH = Path(fd.name)
    
    weights = graph.es["weight"] if weighted and "weight" in graph.es.attributes() else None
    for i, edge in enumerate(graph.es):
        src, tgt = edge.tuple
        if weights:
            fd.write(f"{src} {tgt} {weights[i]}\n")
        else:
            fd.write(f"{src} {tgt}\n")
    fd.close()
    return str(_TEMP_GRAPH_PATH)

def load_graph(
    source: ig.Graph | nx.Graph | str,
    weight_attr: str = "weight",
    default_weight: float = 1.0,
    is_weighted: Optional[bool] = None,
) -> tuple[ig.Graph, bool, str]:
    """
    Load graph from path or in-memory object.
    
    Returns: (igraph.Graph, is_weighted, path_for_workers)
    
    If source is in-memory, writes to temp file so workers can reload independently.
    
    Supported file formats:
        - Edgelist/NCOL (.graph, .edgelist, .txt, etc.): "src tgt [weight]"
        - Adjacency list (.adjlist, .adj)
    
    For other formats (.net, .graphml, .gml), load with igraph/networkx 
    first and pass the graph object directly.
    """
    # NetworkX graph
    if isinstance(source, nx.Graph):
        node_map = {n: i for i, n in enumerate(source.nodes())}
        edges = [(node_map[u], node_map[v]) for u, v in source.edges()]
        graph = ig.Graph(n=len(node_map), edges=edges, directed=source.is_directed())
        if weight_attr:
            graph.es["weight"] = [
                float(source[u][v].get(weight_attr, source[u][v].get("weight", default_weight)))
                for u, v in source.edges()
            ]
        detected = any(weight_attr in d or "weight" in d for _, _, d in source.edges(data=True))
        weighted = detected if is_weighted is None else is_weighted
        if not weighted:
            graph.es["weight"] = [default_weight] * graph.ecount()
        path = _graph_to_temp_file(graph, weighted)
        return graph, weighted, path

    # igraph graph
    if isinstance(source, ig.Graph):
        graph = source.copy()
        detected = "weight" in graph.es.attributes()
        weighted = detected if is_weighted is None else is_weighted
        if not detected or not weighted:
            graph.es["weight"] = [default_weight] * graph.ecount()
        path = _graph_to_temp_file(graph, weighted)
        return graph, weighted, path

    # String path
    path = source
    suffix = Path(path).suffix.lower()
    
    if suffix in {".adjlist", ".adj"}:
        nx_graph = nx.read_adjlist(path)
        return load_graph(nx_graph, weight_attr=weight_attr, default_weight=default_weight, is_weighted=is_weighted)

    detected = _detect_weighted(path)
    weighted = detected if is_weighted is None else is_weighted
    
    if weighted:
        graph = ig.Graph.Read_Ncol(path, weights=True, directed=False)
    else:
        graph = ig.Graph.Read_Ncol(path, weights=False, directed=False)
        graph.es["weight"] = [default_weight] * graph.ecount()
    
    return graph, weighted, path

def load_graph_worker(
    path: str,
    weight_attr: str = "weight", 
    default_weight: float = 1.0,
    is_weighted: bool = False,
) -> ig.Graph:
    """Worker-side graph loading. Simple path-only load."""
    if is_weighted:
        graph = ig.Graph.Read_Ncol(path, weights=True, directed=False)
    else:
        graph = ig.Graph.Read_Ncol(path, weights=False, directed=False)
        graph.es["weight"] = [default_weight] * graph.ecount()
    return graph
