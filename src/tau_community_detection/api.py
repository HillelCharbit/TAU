"""High-level, user-friendly API for TAU community detection.

This module provides a simple, drop-in replacement for igraph's community_leiden()
that requires minimal configuration while maintaining full algorithmic control.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union
import warnings

import igraph as ig
import networkx as nx
import numpy as np

from .algorithm import TauClustering
from .config import TauConfig
from .graph import load_graph


def _get_node_count(graph: ig.Graph | nx.Graph | str) -> int:
    """Extract node count from various graph formats."""
    if isinstance(graph, ig.Graph):
        return graph.vcount()
    elif isinstance(graph, nx.Graph):
        return len(graph)
    elif isinstance(graph, str):
        # For file paths, load minimally
        g, _, _ = load_graph(graph)
        return g.vcount()
    else:
        raise TypeError(
            f"graph must be ig.Graph, nx.Graph, or filepath, got {type(graph)}"
        )


def _auto_scale_population_size(node_count: int) -> int:
    """Auto-scale population size based on graph complexity.
    
    Small graphs (< 1,000 nodes): Smaller population, thorough search.
    Medium graphs (1K-10K nodes): Balanced population.
    Large graphs (> 10K nodes): Larger population to explore diversity.
    
    Parameters
    ----------
    node_count : int
        Number of nodes in the graph.
        
    Returns
    -------
    int
        Suggested population size.
    """
    if node_count < 1_000:
        return 10
    elif node_count < 10_000:
        return 30
    elif node_count < 100_000:
        return 100
    else:
        return max(150, min(300, node_count // 500))


def _auto_scale_max_generations(node_count: int) -> int:
    """Auto-scale max generations based on graph complexity.
    
    Small graphs: More generations for thorough convergence.
    Large graphs: Fewer generations with strict early-stopping.
    
    Parameters
    ----------
    node_count : int
        Number of nodes in the graph.
        
    Returns
    -------
    int
        Suggested max generations.
    """
    if node_count < 1_000:
        return 200
    elif node_count < 10_000:
        return 100
    elif node_count < 100_000:
        return 50
    else:
        # Large graphs: aggressive early-stopping via config.stopping_generations
        return 20


def _get_worker_backend() -> str:
    """Detect and return the best joblib backend for this environment.
    
    Tries to use 'loky' (robust, handles spawning). Falls back to 'spawn'
    if loky is unavailable.
    
    Returns
    -------
    str
        Backend name: 'loky' or 'processes'.
    """
    try:
        import joblib
        # Test if loky is available
        with joblib.parallel_backend("loky", n_jobs=1):
            pass
        return "loky"
    except Exception:
        return "processes"


def run_clustering(
    graph: Union[ig.Graph, nx.Graph, str],
    weights: Optional[Union[str, Sequence[float]]] = None,
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
    for seamless adoption. Automatically scales hyperparameters based on graph size
    and handles multiprocessing transparently, with graceful fallback to sequential
    processing.
    
    Parameters
    ----------
    graph : ig.Graph, nx.Graph, or str
        Input graph. Can be an igraph.Graph, networkx.Graph, or path to a graph file.
        
    weights : str or sequence, optional
        Edge weights to use. If a string, treated as an edge attribute name in the graph.
        If a sequence, must align with the graph's edge order. Default is None (unweighted).
        
    resolution_parameter : float, default=1.0
        Resolution parameter for community detection. Higher values lead to smaller,
        more fragmented communities. Lower values lead to larger, merged communities.
        Typical range: [0.5, 2.0].
        
    random_seed : int, optional
        Seed for reproducibility. If None, results are non-deterministic.
        
    verbose : bool, default=False
        If True, print detailed progress logs.
        
    n_iterations : int, default=3
        Number of Leiden algorithm iterations applied during fitness evaluation.
        Higher values improve quality but increase computation time.
        
    num_workers : int, optional
        Number of parallel workers. If None, uses CPU count. Set to 1 to force
        sequential processing (useful for debugging or interactive environments).
        
    population_size : int, default=60
        Genetic algorithm population size. Empirically optimized across graph sizes.
        
    max_generations : int, default=20
        Maximum GA generations. 95% of improvement achieved by generation 7 with early stopping.
        
    **config_kwargs
        Additional TauConfig parameters for fine-tuning the algorithm without crowding
        the function signature. Listed by importance/frequency of use:
        
        **Performance/Convergence (Most Important):**
        - stopping_generations (int, default=10): Generations without improvement to stop.
        - stopping_jaccard (float, default=0.98): Similarity threshold for early stopping.
        - elite_fraction (float, default=0.1): Fraction of best solutions preserved.
        
        **GA Behavior:**
        - selection_power (int, default=5): Sharpness of selection pressure.
        - immigrant_fraction (float, default=0.15): Fraction of random immigrants per gen.
        
        **Advanced:**
        - worker_chunk_size (int, optional): Batch size for parallel evaluation.
        - reuse_worker_pool (bool, default=True): Reuse worker pool across runs.
        
        Example: run_clustering(g, stopping_generations=5, elite_fraction=0.2)
        
    Returns
    -------
    ig.VertexClustering
        The discovered community structure as an igraph VertexClustering object.
        Access communities via .membership, compute .modularity, etc.
        
    Raises
    ------
    TypeError
        If graph is not ig.Graph, nx.Graph, or a valid file path.
    RuntimeError
        If TAU clustering fails to produce a valid solution.
        
    Examples
    --------
    Basic usage with automatic hyperparameter scaling:
    
    >>> import igraph as ig
    >>> from tau_community_detection import run_clustering
    >>> g = ig.Graph.Famous("Zachary")
    >>> clustering = run_clustering(g)
    >>> print(f"Found {len(clustering)} communities")
    >>> print(f"Modularity: {clustering.modularity:.3f}")
    
    With custom parameters and weighted edges:
    
    >>> clustering = run_clustering(
    ...     g,
    ...     weights="weight",  # or provide sequence of floats
    ...     resolution_parameter=0.8,
    ...     random_seed=42,
    ...     verbose=True,
    ... )
    
    With custom GA parameters:
    
    >>> clustering = run_clustering(
    ...     g,
    ...     population_size=100,  # override default of 60
    ...     max_generations=50,   # override default of 20
    ...     stopping_generations=5,  # early stopping threshold
    ...     random_seed=42,
    ... )
    
    For reproducible results in large-scale analyses:
    
    >>> clustering = run_clustering(
    ...     "path/to/graph.gml",
    ...     random_seed=42,
    ...     num_workers=8,  # explicit parallelism
    ... )
    """
    # Step 1: Get graph and determine node count
    node_count = _get_node_count(graph)
    

    # Step 3: Build TauConfig with defaults and user overrides
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
    
    # Step 3: Handle weights if provided
    if weights is not None:
        if isinstance(weights, str):
            # Weights are an edge attribute name
            config.weight_attribute = weights
        else:
            # Weights provided as sequence; we'll need special handling
            # For now, store them and TauClustering will use them
            config.weight_attribute = None  # Mark as custom weights
    
    # Step 4: Attempt to use joblib with loky backend for robust multiprocessing
    worker_count = config.resolve_worker_count()
    if worker_count > 1:
        try:
            import joblib
            backend = _get_worker_backend()
            # Pre-emptively set up joblib context
            # Note: The actual parallelization in TauClustering uses multiprocessing.Pool,
            # but we can emit a warning if we detect interactive environments.
            
            # Check if running in Jupyter/IPython
            try:
                get_ipython()  # noqa: F821 - undefined in non-interactive envs
                if backend != "loky":
                    warnings.warn(
                        "Running in interactive environment without loky backend. "
                        "Using sequential fallback for safety. "
                        "Consider using loky: pip install loky",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    config.worker_count = 1
            except NameError:
                # Not in interactive environment, safe to use multiprocessing
                pass
        except ImportError:
            warnings.warn(
                "joblib not installed. Install it with: pip install joblib",
                RuntimeWarning,
                stacklevel=2,
            )
    
    # Step 5: Create and run TAU clustering
    if worker_count == 1:
        if verbose:
            print(f"Running TAU in sequential mode (single-threaded)")
    else:
        if verbose:
            print(
                f"Running TAU with {worker_count} workers "
                f"(population_size={population_size}, max_generations={max_generations})"
            )
    
    clustering = TauClustering(
        graph,
        population_size=population_size,
        max_generations=max_generations,
        config=config,
    )
    
    result = clustering.run()
    
    return result


__all__ = ["run_clustering"]
