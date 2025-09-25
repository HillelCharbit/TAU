"""Regression tests on a tiny synthetic graph."""

from __future__ import annotations

import networkx as nx
import numpy as np

from tau_community_detection import TauClustering, TauConfig


def test_small_graph_clustering_runs() -> None:
    """Ensure the GA converges and returns a membership vector on a tiny ER graph."""
    graph = nx.erdos_renyi_graph(24, 0.2, seed=5)

    config = TauConfig(
        population_size=6,
        max_generations=4,
        worker_count=1,
        sim_sample_size=12,
        random_seed=5,
        reuse_worker_pool=False,
    )

    clustering = TauClustering(
        graph,
        population_size=config.population_size,
        max_generation=config.max_generations,
        config=config,
    )
    membership, history = clustering.run()
    clustering.close()

    assert membership.shape == (graph.number_of_nodes(),)
    assert np.issubdtype(membership.dtype, np.integer)
    assert history, "Expected at least one modularity measurement"
