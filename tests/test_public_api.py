"""Regression: bare-default run_clustering must not raise.

Catching the class of bug where run_clustering(g) raises TypeError because
the wrapper's internal kwargs-to-TauConfig merge passes a parameter twice.
These tests call the real TauConfig constructor — no monkeypatching.
"""
import igraph as ig
import networkx as nx
import pytest

from tau_community_detection import run_clustering, TauClustering, TauConfig


@pytest.fixture
def petersen() -> ig.Graph:
    return ig.Graph.Famous("Petersen")


def test_run_clustering_bare_defaults_igraph(petersen):
    result = run_clustering(petersen)
    assert isinstance(result, ig.VertexClustering)
    assert len(result.membership) == petersen.vcount()


def test_run_clustering_bare_defaults_networkx():
    g = nx.petersen_graph()
    result = run_clustering(g)
    assert isinstance(result, ig.VertexClustering)
    assert len(result.membership) == g.number_of_nodes()


def test_run_clustering_named_overrides(petersen):
    result = run_clustering(
        petersen,
        resolution=0.8,
        random_seed=42,
        population_size=20,
        max_generations=5,
    )
    assert isinstance(result, ig.VertexClustering)
