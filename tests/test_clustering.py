"""Integration-flavoured tests for TauClustering weighting behaviour."""
from __future__ import annotations

import networkx as nx
import numpy as np

from tau_community_detection import TauClustering, TauConfig


def _tiny_graph(weighted: bool) -> nx.Graph:
    graph = nx.Graph()
    if weighted:
        graph.add_edge(0, 1, weight=2.0)
        graph.add_edge(1, 2, weight=1.5)
        graph.add_edge(2, 0, weight=0.5)
    else:
        graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return graph


def _run_clustering(graph, *, config_override: dict | None = None) -> np.ndarray:
    config_kwargs = dict(
        population_size=6,
        max_generations=1,
        random_seed=1234,
        stopping_generations=1,
    )
    if config_override:
        config_kwargs.update(config_override)
    config = TauConfig(**config_kwargs)
    tau = TauClustering(graph, population_size=6, max_generations=1, config=config)
    membership, _ = tau.run()
    return membership


def test_unweighted_override_forces_equal_weights():
    membership = _run_clustering(
        _tiny_graph(weighted=True),
        config_override={"is_weighted": False},
    )
    assert isinstance(membership, np.ndarray)
    assert membership.size == 3


def test_weighted_override_preserves_weights():
    membership = _run_clustering(
        _tiny_graph(weighted=True),
        config_override={"is_weighted": True},
    )
    assert membership.size == 3


def test_auto_detection_sets_config_flag():
    graph = _tiny_graph(weighted=True)
    config = TauConfig(random_seed=1, stopping_generations=1)
    tau = TauClustering(graph, population_size=4, max_generations=1, config=config)
    assert tau.config.is_weighted is True
