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
    # Must have >3 columns on at least one line to be detected as adjlist
    path = tmp_path / "unweighted.adjlist"
    path.write_text("0 1 2 3\n1 0 2\n2 0 1\n3 0\n", encoding="utf-8")
    return str(path)


def test_weighted_edgelist_detection(tmp_edgelist):
    graph, resolved, _ = load_graph(tmp_edgelist)
    assert resolved is True
    assert sorted(graph.es["weight"]) == [1.5, 2.5]


def test_adjlist_detection(tmp_adjlist):
    graph, resolved, _ = load_graph(tmp_adjlist)
    assert resolved is False
    assert graph.ecount() >= 3
    assert all(weight == 1.0 for weight in graph.es["weight"])


def test_in_memory_networkx_detection():
    graph = nx.Graph()
    graph.add_edge(1, 2, weight=3.0)
    ig_graph, resolved, _ = load_graph(graph)
    assert resolved is True
    assert ig_graph.es["weight"] == [3.0]


def test_in_memory_override():
    graph = nx.Graph()
    graph.add_edge(1, 2, weight=3.0)
    ig_graph, resolved, _ = load_graph(graph, is_weighted=False)
    assert resolved is False
    assert ig_graph.es["weight"] == [1.0]