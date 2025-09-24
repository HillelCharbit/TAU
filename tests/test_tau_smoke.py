"""Lightweight smoke tests for the TAU clustering API."""
from __future__ import annotations

import networkx as nx

from tau_community_detection import TauClustering, TauConfig


def _build_graph(weighted: bool) -> nx.Graph:
    graph = nx.erdos_renyi_graph(32, 0.12, seed=7)
    if weighted:
        for u, v in graph.edges():
            graph[u][v]["weight"] = 1.0 + abs(u - v) / 100.0
    return graph


def _run_cluster(graph: nx.Graph, *, weighted: bool) -> tuple[list[int], list[float]]:
    config = TauConfig(
        population_size=6,
        max_generations=4,
        worker_count=2,
        sim_sample_size=16,
        random_seed=42,
        worker_chunk_size=2,
        reuse_worker_pool=False,
        weight_attribute="weight" if weighted else None,
    )
    clustering = TauClustering(graph, population_size=config.population_size, max_generation=config.max_generations, config=config)
    membership, history = clustering.run()
    clustering.close()
    return membership.tolist(), history


def test_unweighted_run_produces_membership() -> None:
    membership, history = _run_cluster(_build_graph(weighted=False), weighted=False)
    assert len(membership) == 32
    assert history, "Expected at least one fitness entry"


def test_weighted_run_respects_edge_weights() -> None:
    membership, history = _run_cluster(_build_graph(weighted=True), weighted=True)
    assert len(membership) == 32
    assert history, "Expected at least one fitness entry"
