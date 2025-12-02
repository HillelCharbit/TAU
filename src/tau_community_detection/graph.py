"""Graph loading for TAU."""
from __future__ import annotations

import tempfile
from pathlib import Path

import igraph as ig
import networkx as nx

_TEMP_GRAPH_PATH: Path | None = None


def load_graph(
    source: ig.Graph | nx.Graph | str,
    weight_attr: str = "weight",
    default_weight: float = 1.0,
    is_weighted: bool | None = None,
) -> tuple[ig.Graph, bool, str]:
    """
    Load graph from path or in-memory object.
    
    Returns: (igraph.Graph, is_weighted, path_for_workers)
    
    If source is in-memory, writes to temp file so workers can reload independently.
    
    Supported file formats:
        - Edgelist/NCOL (.graph, .edgelist, .txt, etc.): "src tgt [weight]"
        - Adjacency list (.adjlist, .adj): "src neighbor1 neighbor2 ..."
    
    For other formats (.net, .graphml, .gml), load with igraph/networkx 
    first and pass the graph object directly.
    """
    # Handle in-memory graphs
    if isinstance(source, nx.Graph):
        # Detect weights from original NetworkX graph BEFORE conversion
        detected = any(weight_attr in d or "weight" in d for _, _, d in source.edges(data=True))
        graph = ig.Graph.from_networkx(source)
        
        # Rename custom weight attribute to "weight" for consistency
        if weight_attr != "weight" and weight_attr in graph.es.attributes():
            graph.es["weight"] = graph.es[weight_attr]

    elif isinstance(source, ig.Graph):
        graph = source.copy()
        detected = "weight" in graph.es.attributes()

    else:
        # String path - detect format and load
        if _is_adjlist_format(source):
            nx_graph = nx.read_adjlist(source)
            # Remove isolated nodes that might come from empty lines
            nx_graph.remove_nodes_from(list(nx.isolates(nx_graph)))
            return load_graph(
                nx_graph,
                weight_attr=weight_attr,
                default_weight=default_weight,
                is_weighted=is_weighted,
            )

        detected = _detect_weighted(source)
        weighted = detected if is_weighted is None else is_weighted

        # igraph.Read_Ncol doesn't handle comments - preprocess if needed
        clean_path = _preprocess_edgelist(source)

        try:
            graph = ig.Graph.Read_Ncol(clean_path, weights=detected, directed=False)
        finally:
            # Clean up temp file if we created one
            if clean_path != source:
                Path(clean_path).unlink(missing_ok=True)

        # Apply default weights if needed
        if not detected:
            graph.es["weight"] = [default_weight] * graph.ecount()

        return graph, weighted, source

    # Finalize in-memory graph cases
    weighted = detected if is_weighted is None else is_weighted
    
    # Set weights: keep existing only if detected AND we want weighted
    if not (weighted and detected):
        graph.es["weight"] = [default_weight] * graph.ecount()

    return graph, weighted, _graph_to_temp_file(graph, weighted)

def load_graph_worker(
    path: str,
    default_weight: float = 1.0,
    is_weighted: bool = False,
) -> ig.Graph:
    """Worker-side graph loading. Simple path-only load."""
    graph = ig.Graph.Read_Ncol(path, weights=is_weighted, directed=False)
    if not is_weighted:
        graph.es["weight"] = [default_weight] * graph.ecount()
    return graph

def _is_adjlist_format(path: str) -> bool:
    """Check if file is adjacency list format (>3 columns on first data line)."""
    try:
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    return len(line.split()) > 3
    except Exception:
        pass
    return False

def _detect_weighted(path: str) -> bool:
    """Check if edgelist file has weight column (3rd column is numeric)."""
    try:
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                cols = line.split()
                if len(cols) >= 3:
                    float(cols[2])
                    return True
                return False
    except (OSError, ValueError):
        pass
    return False

def _preprocess_edgelist(path: str) -> str:
    """Remove comments and blank lines from edgelist. Returns path (original or temp)."""
    needs_cleaning = False

    # Check if cleaning is needed
    try:
        with open(path) as f:
            for line in f:
                stripped = line.strip()
                if not stripped or "#" in line:
                    needs_cleaning = True
                    break
    except Exception:
        return path

    if not needs_cleaning:
        return path

    # Create cleaned temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ncol", delete=False) as fd:
        with open(path) as f:
            for line in f:
                clean = line.split("#", 1)[0].strip()
                if clean:
                    fd.write(clean + "\n")
        return fd.name

def _graph_to_temp_file(graph: ig.Graph, weighted: bool) -> str:
    """Write igraph to temp NCOL file, return path."""
    global _TEMP_GRAPH_PATH

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ncol", delete=False) as fd:
        _TEMP_GRAPH_PATH = Path(fd.name)
        weights = graph.es["weight"] if weighted and "weight" in graph.es.attributes() else None

        for i, edge in enumerate(graph.es):
            src, tgt = edge.tuple
            if weights:
                fd.write(f"{src} {tgt} {weights[i]}\n")
            else:
                fd.write(f"{src} {tgt}\n")

    return str(_TEMP_GRAPH_PATH)