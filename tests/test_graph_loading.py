"""Tests for graph loading and weighting detection."""
from __future__ import annotations

import networkx as nx
import pytest

from tau_community_detection.graph import load_graph


@pytest.fixture()
def tmp_edgelist(tmp_path):
    path = tmp_path / "weighted.edgelist"
    path.write_text("0 1 2.5\n1 2 1.5\n", encoding="utf-8")
    return str(path)


@pytest.fixture()
def tmp_adjlist(tmp_path):
    path = tmp_path / "unweighted.adjlist"
    path.write_text("0 1 2\n1 2\n2\n", encoding="utf-8")
    return str(path)


def test_weighted_edgelist_detection(tmp_edgelist):
    graph, resolved = load_graph(tmp_edgelist, return_is_weighted=True)
    assert resolved is True
    assert sorted(graph.es["weight"]) == [1.5, 2.5]


def test_adjlist_detection(tmp_adjlist):
    graph, resolved = load_graph(tmp_adjlist, return_is_weighted=True)
    assert resolved is False
    assert graph.ecount() == 3
    assert all(weight == 1.0 for weight in graph.es["weight"])


def test_in_memory_networkx_detection():
    graph = nx.Graph()
    graph.add_edge(1, 2, weight=3.0)
    ig_graph, resolved = load_graph(graph, return_is_weighted=True)
    assert resolved is True
    assert ig_graph.es["weight"] == [3.0]


def test_in_memory_override():
    graph = nx.Graph()
    graph.add_edge(1, 2, weight=3.0)
    ig_graph, resolved = load_graph(graph, return_is_weighted=True, is_weighted=False)
    assert resolved is False
    assert ig_graph.es["weight"] == [1.0]
