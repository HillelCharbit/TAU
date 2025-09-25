"""Lean integration tests for the TAU clustering pipeline.

These tests exercise a tiny Erdős-Rényi graph to keep runtime low while still
covering the public API. The aim is to document how the components fit
together rather than to assert specific modularity values (which are highly
sensitive to randomness).
"""
from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np

from tau_community_detection import TauClustering, TauConfig


def _configure(seed: int, *, weighted: bool) -> tuple[TauConfig, nx.Graph]:
    """Create a deterministic graph and TauConfig for testing.

    Parameters
    ----------
    seed:
        Controls both the NetworkX graph and TAU's RNG so observations are
        reproducible.
    weighted:
        When true, assigns a synthetic weight to every edge so we validate the
        weighted path.
    """
    graph = nx.erdos_renyi_graph(30, 0.15, seed=seed)
    if weighted:
        for u, v in graph.edges():
            graph[u][v]["weight"] = 1.0 + abs(u - v) / 100.0

    config = TauConfig(
        population_size=4,
        max_generations=3,
        worker_count=1,
        sim_sample_size=12,
        random_seed=seed,
        worker_chunk_size=2,
        reuse_worker_pool=False,
        weight_attribute="weight" if weighted else None,
    )
    return config, graph


def _run_tau(config: TauConfig, graph: nx.Graph) -> tuple[np.ndarray, Iterable[float]]:
    """Run TAU end-to-end and return the membership vector + fitness history."""
    clustering = TauClustering(
        graph,
        population_size=config.population_size,
        max_generation=config.max_generations,
        config=config,
    )
    membership, history = clustering.run()
    clustering.close()
    return membership, history


class TestTauClustering:
    """Grouped tests for clarity when reading pytest output."""

    def test_unweighted_clustering_produces_partition(self) -> None:
        """TAU should produce a membership array for an unweighted graph."""
        config, graph = _configure(seed=7, weighted=False)
        membership, history = _run_tau(config, graph)

        assert membership.shape == (graph.number_of_nodes(),)
        assert membership.dtype == np.int64
        assert history, "Expect at least one modularity measurement"

    def test_weighted_clustering_respects_weights(self) -> None:
        """Ensure the weighted configuration runs without dropping edge weights."""
        config, graph = _configure(seed=11, weighted=True)
        membership, history = _run_tau(config, graph)

        assert membership.shape == (graph.number_of_nodes(),)
        assert history[-1] >= min(history)

    def test_repeated_runs_are_reproducible(self) -> None:
        """With the same seed the clustering output should be identical."""
        config, graph = _configure(seed=21, weighted=False)

        first_membership, first_history = _run_tau(config, graph)
        second_membership, second_history = _run_tau(config, graph)

        np.testing.assert_array_equal(first_membership, second_membership)
        np.testing.assert_allclose(first_history, second_history, rtol=1e-2, atol=1e-4)
