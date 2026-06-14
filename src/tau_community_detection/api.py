"""High-level, user-friendly API for TAU community detection.

This module provides a simple, drop-in replacement for igraph's community_leiden()
that requires minimal configuration while maintaining full algorithmic control.
"""
from __future__ import annotations

from typing import Optional, Union
import warnings

import igraph as ig
import networkx as nx

from .algorithm import TauClustering
from .config import TauConfig


def run_clustering(
    graph: Union[ig.Graph, nx.Graph, str],
    population_size: int = 60,
    max_generations: int = 20,
    resolution_parameter: float = 1.0,
    random_seed: Optional[int] = None,
    verbose: bool = False,
    n_iterations: int = 3,
    num_workers: Optional[int] = None,
    **config_kwargs,
) -> ig.VertexClustering:
    """Find community structure using TAU evolutionary clustering.

    A high-level, user-friendly interface that mimics igraph's community_leiden()
    for seamless adoption. Automatically handles multiprocessing with graceful
    fallback to sequential processing in interactive environments.

    Parameters
    ----------
    graph : ig.Graph, nx.Graph, or str
        Input graph. Can be an igraph.Graph, networkx.Graph, or path to a graph file.
        Edge weights are detected automatically. To override, pass
        ``TauConfig(is_weighted=True/False)`` via ``**config_kwargs``.

    resolution_parameter : float, default=1.0
        Resolution parameter for community detection. Higher values lead to smaller,
        more fragmented communities. Typical range: [0.5, 2.0].

    random_seed : int, optional
        Seed for reproducibility. If None, results are non-deterministic.

    verbose : bool, default=False
        If True, print detailed progress logs.

    n_iterations : int, default=3
        Number of Leiden algorithm iterations per fitness evaluation.

    num_workers : int, optional
        Number of parallel workers. If None, uses CPU count. Set to 1 to force
        sequential processing (useful for debugging or interactive environments).

    population_size : int, default=60
        Genetic algorithm population size.

    max_generations : int, default=20
        Maximum GA generations.

    **config_kwargs
        Additional TauConfig parameters for fine-tuning without crowding the
        function signature. Common overrides:

        - stopping_generations (int, default=10): Generations without improvement to stop.
        - stopping_jaccard (float, default=0.98): Similarity threshold for early stopping.
        - elite_fraction (float, default=0.1): Fraction of best solutions preserved.
        - selection_power (int, default=5): Sharpness of selection pressure.
        - immigrant_fraction (float, default=0.15): Fraction of random immigrants per gen.

    Returns
    -------
    ig.VertexClustering
        The discovered community structure. Access communities via .membership,
        compute .modularity, etc.

    Raises
    ------
    TypeError
        If graph is not ig.Graph, nx.Graph, or a valid file path.
    ValueError
        If any hyperparameter is out of its valid range.
    RuntimeError
        If TAU clustering fails to produce a valid solution.

    Examples
    --------
    Basic usage:

    >>> import igraph as ig
    >>> from tau_community_detection import run_clustering
    >>> g = ig.Graph.Famous("Zachary")
    >>> clustering = run_clustering(g)
    >>> print(f"Found {len(clustering)} communities, modularity={clustering.modularity:.3f}")

    With custom parameters:

    >>> clustering = run_clustering(
    ...     g,
    ...     resolution_parameter=0.8,
    ...     random_seed=42,
    ...     verbose=True,
    ...     stopping_generations=5,
    ... )
    """
    # Step 1: Warn if running in Jupyter/IPython without loky
    try:
        get_ipython()  # noqa: F821 - only defined in interactive environments
        try:
            from loky.process_executor import LokyProcessPoolExecutor  # noqa: F401
        except ImportError:
            warnings.warn(
                "Running in an interactive environment without loky. "
                "Falling back to sequential processing. "
                "Install loky for parallel support: pip install loky",
                RuntimeWarning,
                stacklevel=2,
            )
            config_kwargs.setdefault("worker_count", 1)
    except NameError:
        pass  # not in an interactive environment

    # Step 2: Build TauConfig
    config = TauConfig(
        population_size=population_size,
        max_generations=max_generations,
        worker_count=num_workers,
        n_iterations=n_iterations,
        resolution_parameter=resolution_parameter,
        random_seed=random_seed,
        verbose=verbose,
        **config_kwargs,
    )

    # Step 3: Run clustering
    return TauClustering(
        graph,
        population_size=population_size,
        max_generations=max_generations,
        config=config,
    ).run()


__all__ = ["run_clustering"]
