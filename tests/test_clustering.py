"""Integration-flavoured tests for TauClustering weighting behaviour."""
from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pytest

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


def _run_clustering(graph, **kwargs):
    config = TauConfig(
        population_size=6,
        max_generations=1,
        random_seed=1234,
        stopping_generations=1,
    )
    tau = TauClustering(graph, population_size=6, max_generations=1, config=config, **kwargs)
    membership, _ = tau.run()
    return membership


def test_unweighted_override_forces_equal_weights():
    membership = _run_clustering(_tiny_graph(weighted=True), is_weighted=False)
    assert isinstance(membership, np.ndarray)
    assert membership.size == 3


def test_weighted_override_preserves_weights():
    membership = _run_clustering(_tiny_graph(weighted=True), is_weighted=True)
    assert membership.size == 3


def test_config_constructor_conflict_warns():
    graph = _tiny_graph(weighted=True)
    config = TauConfig(is_weighted=True, random_seed=1, stopping_generations=1)
    with warnings.catch_warnings(record=True) as caught:
        TauClustering(
            graph,
            population_size=4,
            max_generations=1,
            config=config,
            is_weighted=False,
        )
        assert any("Mismatch" in str(w.message) for w in caught)
