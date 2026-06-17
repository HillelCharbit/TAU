"""High-level, user-friendly API for TAU community detection.

This module provides a simple, drop-in replacement for igraph's community_leiden()
that requires minimal configuration while maintaining full algorithmic control.
"""
from __future__ import annotations

from typing import Optional, Union

import igraph as ig
import networkx as nx

from .algorithm import TauClustering
from .config import TauConfig


def run_clustering(
    graph: Union[ig.Graph, nx.Graph, str],
    population_size: int = 60,
    max_generations: int = 20,
    resolution: float = 1.0,
    random_seed: Optional[int] = None,
    verbose: bool = False,
    n_iterations: int = 3,
    worker_count: Optional[int] = None,
) -> ig.VertexClustering:
    """Find community structure using TAU evolutionary clustering.

    A high-level, user-friendly interface that mimics igraph's community_leiden()
    for seamless adoption. Automatically handles multiprocessing with graceful
    fallback to sequential processing in interactive environments.

    For fine-grained control over the genetic algorithm internals, use
    ``TauClustering`` with a ``TauConfig`` directly.

    Parameters
    ----------
    graph : ig.Graph, nx.Graph, or str
        Input graph. Can be an igraph.Graph, networkx.Graph, or path to a graph file.
        Edge weights are detected automatically.

    population_size : int, default=60
        Genetic algorithm population size.

    max_generations : int, default=20
        Maximum GA generations.

    resolution : float, default=1.0
        Resolution parameter for community detection. Higher values lead to smaller,
        more fragmented communities. Typical range: [0.5, 2.0].

    random_seed : int, optional
        Integer seed for the random number generator. For reproducible results,
        also fix ``worker_count``; leaving it automatic ties results to
        ``cpu_count()`` and may differ across machines.

    verbose : bool, default=False
        If True, print detailed progress logs.

    n_iterations : int, default=3
        Number of Leiden algorithm iterations per fitness evaluation.

    worker_count : int, optional
        Number of parallel workers. If None, defaults to cpu_count() capped by
        population_size + immigrant_count (via TauConfig). Set to 1 to force
        sequential processing (useful for debugging or interactive environments).

    Returns
    -------
    ig.VertexClustering
        The discovered community structure. Access communities via .membership,
        compute .modularity, etc.

    Raises
    ------
    TypeError
        If graph is not an ig.Graph, nx.Graph, or str.
    FileNotFoundError
        If graph is a path string that does not point to a file.
    ValueError
        If any hyperparameter is out of its valid range.
    RuntimeError
        Defensive guard; not reachable under normal configuration.

    Examples
    --------
    Basic usage:

    >>> import igraph as ig
    >>> from tau_community_detection import run_clustering
    >>> g = ig.Graph.Famous("Zachary")
    >>> clustering = run_clustering(g)
    >>> print(f"Found {len(clustering)} communities, modularity={clustering.modularity:.3f}")

    With custom parameters:

    >>> clustering = run_clustering(g, resolution=0.8, random_seed=42, verbose=True)
    """
    config = TauConfig(
        population_size=population_size,
        max_generations=max_generations,
        n_iterations=n_iterations,
        resolution=resolution,
        random_seed=random_seed,
        verbose=verbose,
        worker_count=worker_count,
    )

    return TauClustering(graph, config=config).run()


__all__ = ["run_clustering"]
