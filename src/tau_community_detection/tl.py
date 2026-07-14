from __future__ import annotations

import igraph as ig
import pandas as pd
from anndata import AnnData
from scipy import sparse

from .api import run_clustering


def _get_connectivities(
    adata: AnnData,
    *,
    neighbors_key: str | None = None,
    obsp: str | None = None,
):
    """Return a connectivity matrix stored in an AnnData object."""

    if neighbors_key is not None and obsp is not None:
        raise ValueError("Specify only one of `neighbors_key` or `obsp`.")

    if obsp is not None:
        connectivities_key = obsp
    elif neighbors_key is not None:
        try:
            connectivities_key = adata.uns[neighbors_key]["connectivities_key"]
        except KeyError as exc:
            raise KeyError(
                f"No connectivity graph found for "
                f"neighbors_key={neighbors_key!r}."
            ) from exc
    else:
        connectivities_key = "connectivities"

    if connectivities_key not in adata.obsp:
        raise KeyError(
            f"{connectivities_key!r} was not found in `adata.obsp`. "
            "Run `scanpy.pp.neighbors(adata)` first."
        )

    return adata.obsp[connectivities_key]


def _connectivities_to_igraph(
    connectivities,
    *,
    use_weights: bool = True,
) -> ig.Graph:
    """Convert a Scanpy connectivity matrix to an undirected igraph graph."""

    matrix = sparse.csr_matrix(connectivities)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The connectivity matrix must be square.")

    # Keep only the upper triangle so each undirected edge appears once.
    upper = sparse.triu(matrix, k=1).tocoo()
    upper.eliminate_zeros()

    edges = list(
        zip(
            upper.row.tolist(),
            upper.col.tolist(),
            strict=True,
        )
    )

    graph = ig.Graph(
        n=matrix.shape[0],
        edges=edges,
        directed=False,
    )

    if use_weights:
        graph.es["weight"] = upper.data.astype(float).tolist()

    graph.vs["name"] = [str(index) for index in range(matrix.shape[0])]

    return graph


def tau(
    adata: AnnData,
    resolution: float = 1.0,
    *,
    random_state: int | None = 0,
    key_added: str = "tau",
    adjacency=None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    use_weights: bool = True,
    copy: bool = False,
    population_size: int = 60,
    max_generations: int = 20,
    n_iterations: int = 3,
    worker_count: int | None = None,
    verbose: bool = False,
) -> AnnData | None:
    """Cluster cells using TAU community detection.

    The function uses a Scanpy neighbor graph, runs TAU clustering, and stores
    the resulting cluster labels in ``adata.obs[key_added]``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    resolution
        Leiden resolution used during TAU fitness evaluation.
    random_state
        Random seed used by TAU.
    key_added
        Key under which cluster labels are stored in ``adata.obs``.
    adjacency
        Optional adjacency matrix. By default, use Scanpy connectivities.
    neighbors_key
        Key in ``adata.uns`` containing neighbor graph metadata.
    obsp
        Key in ``adata.obsp`` containing a connectivity matrix.
    use_weights
        Whether to use connectivity values as edge weights.
    copy
        If True, return a copy of ``adata`` instead of modifying it in place.
    population_size
        Number of candidate partitions per generation.
    max_generations
        Maximum number of genetic algorithm generations.
    n_iterations
        Number of Leiden iterations per fitness evaluation.
    worker_count
        Number of parallel workers. Use 1 for sequential execution.
    verbose
        Print TAU progress information.

    Returns
    -------
    AnnData or None
        Returns a modified copy when ``copy=True``; otherwise returns ``None``.
    """

    if adjacency is not None and (
        neighbors_key is not None or obsp is not None
    ):
        raise ValueError(
            "Do not specify `neighbors_key` or `obsp` when "
            "`adjacency` is provided."
        )

    if copy:
        adata = adata.copy()

    if adjacency is None:
        adjacency = _get_connectivities(
            adata,
            neighbors_key=neighbors_key,
            obsp=obsp,
        )

    graph = _connectivities_to_igraph(
        adjacency,
        use_weights=use_weights,
    )

    clustering = run_clustering(
        graph,
        resolution=resolution,
        random_seed=random_state,
        population_size=population_size,
        max_generations=max_generations,
        n_iterations=n_iterations,
        worker_count=worker_count,
        verbose=verbose,
    )

    adata.obs[key_added] = pd.Categorical(
        [str(label) for label in clustering.membership]
    )

    adata.uns[key_added] = {
        "params": {
            "resolution": resolution,
            "random_state": random_state,
            "population_size": population_size,
            "max_generations": max_generations,
            "n_iterations": n_iterations,
            "worker_count": worker_count,
            "use_weights": use_weights,
        },
        "modularity": float(clustering.modularity),
    }

    return adata if copy else None