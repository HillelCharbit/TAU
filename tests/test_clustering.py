"""Integration-flavoured tests for TauClustering weighting behaviour."""
from __future__ import annotations

import sys

import igraph as ig
import networkx as nx
import numpy as np
import pytest

import tau_community_detection as tau
import tau_community_detection.api as api
from tau_community_detection import TauClustering, TauConfig
from tau_community_detection.partition import Partition, configure_main


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
    tau = TauClustering(graph, config=TauConfig(**config_kwargs))
    vertex_clustering = tau.run()
    return np.asarray(vertex_clustering.membership)


def test_run_clustering_interactive_fallback_does_not_duplicate_worker_count(monkeypatch):
    # api.py passes worker_count through unchanged; verify it does not inject a
    # duplicate kwarg that would cause TauConfig to receive it twice.
    graph = nx.path_graph(4)
    captured = {}

    monkeypatch.setitem(sys.modules, "loky", None)

    class _StubConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _StubClustering:
        def __init__(self, graph, config):
            self.graph = graph
            self.config = config

        def run(self):
            return "ok"

    monkeypatch.setattr(api, "TauConfig", _StubConfig)
    monkeypatch.setattr(api, "TauClustering", _StubClustering)

    assert api.run_clustering(graph) == "ok"
    # worker_count flows through to TauConfig as-is (None = use TauConfig default)
    assert captured.get("worker_count") is None


def test_merge_contiguity_when_comm2_is_last():
    # Graph where every edge crosses from community 0 into community 2 (=last).
    # Edge tuples are (0,4) and (1,4), so comm1=0, comm2=2=last on every pick —
    # the branch the old code skipped, leaving a gap in the label range.
    g = ig.Graph()
    g.add_vertices(5)
    g.add_edges([(0, 4), (1, 4)])
    configure_main(g, n_iterations=1, resolution=1.0, seed=0)

    membership = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    part = Partition.from_membership(membership.copy(), sample_fraction=0.5, n_comms=3)
    part._merge_connected_communities(g, part.membership, np.random.default_rng(0))

    m = part.membership
    assert set(m) == set(range(m.max() + 1)), f"Gap in labels after merge: {sorted(set(m))}"
    assert part.n_comms == m.max() + 1


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
    config = TauConfig(population_size=4, max_generations=1, random_seed=1, stopping_generations=1)
    tau = TauClustering(graph, config=config)
    assert tau.config.is_weighted is True


def test_original_membership_preserves_vertex_names(tmp_path):
    graph_path = tmp_path / "preserve_names.graph"
    graph_path.write_text(
        "\n".join(
            [
                "12 851979 0.5",
                "851979 113878 0.5",
                "113878 12 0.5",
            ]
        )
    )
    config = TauConfig(
        population_size=4,
        max_generations=1,
        random_seed=7,
        stopping_generations=1,
        worker_count=1,
    )
    tau = TauClustering(str(graph_path), config=config)
    clustering = tau.run()

    assert hasattr(clustering, "original_membership")
    names = list(tau.graph.vs["name"])
    assert set(clustering.original_membership.keys()) == set(names)
    for idx, name in enumerate(names):
        assert clustering.original_membership[name] == clustering.membership[idx]
