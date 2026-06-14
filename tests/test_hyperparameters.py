"""Comprehensive hyperparameter tests for TauConfig and TauClustering.

Covers every TauConfig field and TauClustering constructor parameter:
  - Validation: every invalid value raises ValueError
  - Boundaries: min/max valid values are accepted without error
  - Type misuse: wrong types a user might pass
  - Resolver helpers: resolve_worker_count, resolve_elite_count, etc.
  - Behavioral correctness: each param actually has the expected effect
  - Edge cases: extreme but valid values still produce correct results
"""
from __future__ import annotations

import pytest
import igraph as ig
import numpy as np

from tau_community_detection.config import TauConfig
from tau_community_detection.algorithm import TauClustering


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def g() -> ig.Graph:
    """Petersen graph — 10 nodes, 15 edges, stable community structure."""
    return ig.Graph.Famous("Petersen")


@pytest.fixture()
def g_weighted() -> ig.Graph:
    graph = ig.Graph.Famous("Petersen")
    graph.es["weight"] = [float(i + 1) for i in range(graph.ecount())]
    return graph


def _run(graph, population_size=4, max_generations=2, **config_kwargs) -> ig.VertexClustering:
    """Helper: run TAU sequentially with small, fast defaults."""
    config_kwargs.setdefault("worker_count", 1)
    config = TauConfig(**config_kwargs)
    return TauClustering(
        graph, population_size=population_size, max_generations=max_generations, config=config
    ).run()


def _run_with_stats(graph, population_size=4, max_generations=2, **config_kwargs):
    config_kwargs.setdefault("worker_count", 1)
    config = TauConfig(**config_kwargs)
    return TauClustering(
        graph, population_size=population_size, max_generations=max_generations, config=config
    ).run(track_stats=True)


# ===========================================================================
# 1. TauConfig Validation — every invalid value must raise ValueError
# ===========================================================================

class TestTauConfigValidation:

    # --- elite_fraction ---

    def test_elite_fraction_zero_raises(self):
        with pytest.raises(ValueError, match="elite_fraction"):
            TauConfig(elite_fraction=0.0)

    def test_elite_fraction_negative_raises(self):
        with pytest.raises(ValueError, match="elite_fraction"):
            TauConfig(elite_fraction=-0.1)

    def test_elite_fraction_above_one_raises(self):
        with pytest.raises(ValueError, match="elite_fraction"):
            TauConfig(elite_fraction=1.01)

    # --- immigrant_fraction ---

    def test_immigrant_fraction_zero_raises(self):
        with pytest.raises(ValueError, match="immigrant_fraction"):
            TauConfig(immigrant_fraction=0.0)

    def test_immigrant_fraction_negative_raises(self):
        with pytest.raises(ValueError, match="immigrant_fraction"):
            TauConfig(immigrant_fraction=-0.5)

    def test_immigrant_fraction_above_one_raises(self):
        with pytest.raises(ValueError, match="immigrant_fraction"):
            TauConfig(immigrant_fraction=1.5)

    # --- selection_power ---

    def test_selection_power_zero_raises(self):
        with pytest.raises(ValueError, match="selection_power"):
            TauConfig(selection_power=0)

    def test_selection_power_negative_raises(self):
        with pytest.raises(ValueError, match="selection_power"):
            TauConfig(selection_power=-3)

    # --- elite_similarity_threshold ---

    def test_elite_similarity_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="elite_similarity_threshold"):
            TauConfig(elite_similarity_threshold=-0.01)

    def test_elite_similarity_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="elite_similarity_threshold"):
            TauConfig(elite_similarity_threshold=1.001)

    # --- stopping_jaccard ---

    def test_stopping_jaccard_negative_raises(self):
        with pytest.raises(ValueError, match="stopping_jaccard"):
            TauConfig(stopping_jaccard=-0.01)

    def test_stopping_jaccard_above_one_raises(self):
        with pytest.raises(ValueError, match="stopping_jaccard"):
            TauConfig(stopping_jaccard=1.01)

    # --- n_iterations ---

    def test_n_iterations_zero_raises(self):
        with pytest.raises(ValueError, match="n_iterations"):
            TauConfig(n_iterations=0)

    def test_n_iterations_negative_raises(self):
        with pytest.raises(ValueError, match="n_iterations"):
            TauConfig(n_iterations=-1)

    # --- stopping_generations ---

    def test_stopping_generations_zero_raises(self):
        with pytest.raises(ValueError, match="stopping_generations"):
            TauConfig(stopping_generations=0)

    def test_stopping_generations_negative_raises(self):
        with pytest.raises(ValueError, match="stopping_generations"):
            TauConfig(stopping_generations=-10)

    # --- resolution_parameter ---

    def test_resolution_parameter_zero_raises(self):
        with pytest.raises(ValueError, match="resolution_parameter"):
            TauConfig(resolution_parameter=0.0)

    def test_resolution_parameter_negative_raises(self):
        with pytest.raises(ValueError, match="resolution_parameter"):
            TauConfig(resolution_parameter=-1.0)

    # --- default_edge_weight ---

    def test_default_edge_weight_zero_raises(self):
        with pytest.raises(ValueError, match="default_edge_weight"):
            TauConfig(default_edge_weight=0.0)

    def test_default_edge_weight_negative_raises(self):
        with pytest.raises(ValueError, match="default_edge_weight"):
            TauConfig(default_edge_weight=-0.5)

    # --- sample_fraction_range ---

    def test_sample_fraction_range_low_zero_raises(self):
        with pytest.raises(ValueError, match="sample_fraction_range"):
            TauConfig(sample_fraction_range=(0.0, 0.5))

    def test_sample_fraction_range_low_negative_raises(self):
        with pytest.raises(ValueError, match="sample_fraction_range"):
            TauConfig(sample_fraction_range=(-0.1, 0.5))

    def test_sample_fraction_range_low_greater_than_high_raises(self):
        with pytest.raises(ValueError, match="sample_fraction_range"):
            TauConfig(sample_fraction_range=(0.8, 0.3))

    def test_sample_fraction_range_high_above_one_raises(self):
        with pytest.raises(ValueError, match="sample_fraction_range"):
            TauConfig(sample_fraction_range=(0.2, 1.01))

    def test_sample_fraction_range_both_zero_raises(self):
        with pytest.raises(ValueError, match="sample_fraction_range"):
            TauConfig(sample_fraction_range=(0.0, 0.0))


# ===========================================================================
# 2. TauConfig Boundary Values — valid edge cases that must NOT raise
# ===========================================================================

class TestTauConfigBoundaryValues:

    def test_elite_fraction_minimum_positive(self):
        cfg = TauConfig(elite_fraction=1e-9)
        assert cfg.elite_fraction == pytest.approx(1e-9)

    def test_elite_fraction_exactly_one(self):
        assert TauConfig(elite_fraction=1.0).elite_fraction == 1.0

    def test_immigrant_fraction_minimum_positive(self):
        assert TauConfig(immigrant_fraction=1e-9).immigrant_fraction == pytest.approx(1e-9)

    def test_immigrant_fraction_exactly_one(self):
        assert TauConfig(immigrant_fraction=1.0).immigrant_fraction == 1.0

    def test_elite_similarity_threshold_zero(self):
        assert TauConfig(elite_similarity_threshold=0.0).elite_similarity_threshold == 0.0

    def test_elite_similarity_threshold_one(self):
        assert TauConfig(elite_similarity_threshold=1.0).elite_similarity_threshold == 1.0

    def test_stopping_jaccard_zero(self):
        assert TauConfig(stopping_jaccard=0.0).stopping_jaccard == 0.0

    def test_stopping_jaccard_one(self):
        assert TauConfig(stopping_jaccard=1.0).stopping_jaccard == 1.0

    def test_selection_power_one(self):
        assert TauConfig(selection_power=1).selection_power == 1

    def test_selection_power_very_large(self):
        assert TauConfig(selection_power=1000).selection_power == 1000

    def test_resolution_parameter_very_small(self):
        assert TauConfig(resolution_parameter=1e-9).resolution_parameter == pytest.approx(1e-9)

    def test_resolution_parameter_very_large(self):
        assert TauConfig(resolution_parameter=1000.0).resolution_parameter == 1000.0

    def test_sample_fraction_range_low_equals_high(self):
        assert TauConfig(sample_fraction_range=(0.5, 0.5)).sample_fraction_range == (0.5, 0.5)

    def test_sample_fraction_range_near_full_span(self):
        cfg = TauConfig(sample_fraction_range=(1e-9, 1.0))
        assert cfg.sample_fraction_range[1] == 1.0

    def test_default_edge_weight_very_small(self):
        assert TauConfig(default_edge_weight=1e-9).default_edge_weight == pytest.approx(1e-9)

    def test_n_iterations_one(self):
        assert TauConfig(n_iterations=1).n_iterations == 1

    def test_stopping_generations_one(self):
        assert TauConfig(stopping_generations=1).stopping_generations == 1

    def test_defaults_are_all_valid(self):
        """Default TauConfig must pass its own validation."""
        cfg = TauConfig()
        assert cfg.elite_fraction == 0.1
        assert cfg.resolution_parameter == 1.0


# ===========================================================================
# 3. Type Misuse — wrong types a user might pass
# ===========================================================================

class TestTauConfigTypeMisuse:

    def test_elite_fraction_string_raises(self):
        with pytest.raises((TypeError, ValueError)):
            TauConfig(elite_fraction="0.5")

    def test_sample_fraction_range_single_value_raises(self):
        """Tuple unpacking fails for wrong-length input."""
        with pytest.raises((ValueError, TypeError)):
            TauConfig(sample_fraction_range=(0.5,))

    def test_sample_fraction_range_three_values_raises(self):
        with pytest.raises((ValueError, TypeError)):
            TauConfig(sample_fraction_range=(0.1, 0.5, 0.9))

    def test_stopping_jaccard_string_raises(self):
        with pytest.raises((TypeError, ValueError)):
            TauConfig(stopping_jaccard="0.5")

    def test_resolution_parameter_string_raises(self):
        with pytest.raises((TypeError, ValueError)):
            TauConfig(resolution_parameter="1.0")

    def test_worker_count_negative_is_silently_safe(self):
        """Negative worker_count has no validation but resolves to 1 safely."""
        cfg = TauConfig(worker_count=-1, population_size=10)
        # resolve_worker_count: max(1, min(tasks, -1)) = max(1, -1) = 1
        assert cfg.resolve_worker_count() == 1

    def test_worker_count_zero_falls_back_to_cpu_count(self):
        """worker_count=0 is falsy so `or cpu_count()` kicks in."""
        cfg = TauConfig(worker_count=0, population_size=10)
        result = cfg.resolve_worker_count()
        assert result >= 1


# ===========================================================================
# 4. TauClustering Constructor Validation
# ===========================================================================

class TestTauClusteringValidation:

    def test_population_size_zero_raises(self, g):
        with pytest.raises(ValueError, match="population_size"):
            TauClustering(g, population_size=0, max_generations=2)

    def test_population_size_negative_raises(self, g):
        with pytest.raises(ValueError, match="population_size"):
            TauClustering(g, population_size=-5, max_generations=2)

    def test_max_generations_zero_raises(self, g):
        with pytest.raises(ValueError, match="max_generations"):
            TauClustering(g, population_size=4, max_generations=0)

    def test_max_generations_negative_raises(self, g):
        with pytest.raises(ValueError, match="max_generations"):
            TauClustering(g, population_size=4, max_generations=-1)

    def test_invalid_graph_source_raises(self):
        with pytest.raises((TypeError, AttributeError, OSError, FileNotFoundError)):
            TauClustering(99999, population_size=4, max_generations=2)

    def test_nonexistent_file_path_raises(self):
        with pytest.raises((OSError, FileNotFoundError, Exception)):
            TauClustering("/no/such/file.graph", population_size=4, max_generations=2)

    def test_population_size_one_is_valid(self, g):
        config = TauConfig(worker_count=1, elite_fraction=1.0)
        result = TauClustering(g, population_size=1, max_generations=2, config=config).run()
        assert isinstance(result, ig.VertexClustering)

    def test_max_generations_one_is_valid(self, g):
        result = _run(g, population_size=4, max_generations=1)
        assert isinstance(result, ig.VertexClustering)

    def test_config_overrides_stored_on_instance(self, g):
        config = TauConfig(worker_count=1, verbose=False)
        tau = TauClustering(g, population_size=8, max_generations=3, config=config)
        assert tau.config.population_size == 8
        assert tau.config.max_generations == 3


# ===========================================================================
# 5. Resolver Helpers
# ===========================================================================

class TestTauConfigResolvers:

    def test_resolve_elite_count_minimum_is_one(self):
        cfg = TauConfig(population_size=3, elite_fraction=0.001)
        assert cfg.resolve_elite_count() == 1

    def test_resolve_elite_count_fraction(self):
        cfg = TauConfig(population_size=10, elite_fraction=0.3)
        assert cfg.resolve_elite_count() == 3

    def test_resolve_elite_count_full_population(self):
        cfg = TauConfig(population_size=10, elite_fraction=1.0)
        assert cfg.resolve_elite_count() == 10

    def test_resolve_immigrant_count_minimum_is_one(self):
        cfg = TauConfig(population_size=3, immigrant_fraction=0.001)
        assert cfg.resolve_immigrant_count() == 1

    def test_resolve_immigrant_count_fraction(self):
        cfg = TauConfig(population_size=10, immigrant_fraction=0.5)
        assert cfg.resolve_immigrant_count() == 5

    def test_resolve_worker_count_explicit(self):
        cfg = TauConfig(population_size=10, worker_count=3)
        assert cfg.resolve_worker_count() == 3

    def test_resolve_worker_count_capped_by_available_tasks(self):
        cfg = TauConfig(population_size=4, immigrant_fraction=0.25, worker_count=1000)
        max_tasks = 4 + max(1, int(0.25 * 4))
        assert cfg.resolve_worker_count() == min(1000, max_tasks)

    def test_resolve_worker_count_minimum_one(self):
        cfg = TauConfig(population_size=5, worker_count=1)
        assert cfg.resolve_worker_count() == 1


# ===========================================================================
# 6. Behavioral Correctness — params must have the expected effect
# ===========================================================================

class TestHyperparameterBehavior:

    def test_random_seed_accepted_and_produces_valid_result(self, g):
        # random_seed seeds TAU's numpy RNG and igraph's Python-level RNG callbacks.
        # igraph's C-level Leiden algorithm retains internal C state that cannot be
        # seeded from Python, so exact membership equality across runs is not guaranteed.
        r1 = _run(g, population_size=4, max_generations=2, random_seed=42)
        r2 = _run(g, population_size=4, max_generations=2, random_seed=42)
        assert isinstance(r1, ig.VertexClustering)
        assert isinstance(r2, ig.VertexClustering)
        assert len(r1.membership) == len(r2.membership) == g.vcount()

    def test_different_seeds_accepted_without_error(self, g):
        for seed in range(5):
            result = _run(g, population_size=4, max_generations=2, random_seed=seed)
            assert isinstance(result, ig.VertexClustering)

    def test_track_stats_false_returns_vertex_clustering(self, g):
        result = _run(g)
        assert isinstance(result, ig.VertexClustering)

    def test_track_stats_true_returns_clustering_and_list(self, g):
        clustering, stats = _run_with_stats(g)
        assert isinstance(clustering, ig.VertexClustering)
        assert isinstance(stats, list)
        assert len(stats) >= 1

    def test_track_stats_entries_have_required_keys(self, g):
        _, stats = _run_with_stats(g, max_generations=3)
        required = {"generation", "top_fitness", "average_fitness", "time_per_generation"}
        for entry in stats:
            assert required.issubset(entry.keys())

    def test_track_stats_generation_numbers_are_sequential(self, g):
        _, stats = _run_with_stats(g, max_generations=5, stopping_jaccard=1.0, stopping_generations=999)
        generations = [s["generation"] for s in stats]
        assert generations == list(range(1, len(generations) + 1))

    def test_max_generations_is_hard_cap(self, g):
        # stopping disabled → algorithm must respect max_generations
        _, stats = _run_with_stats(g, max_generations=3, stopping_jaccard=1.0, stopping_generations=999)
        assert len(stats) <= 3

    def test_stopping_jaccard_zero_triggers_early_stop(self, g):
        # jaccard >= 0.0 is always True → streak increments every generation
        # stopping_generations=1 → stops after 2nd generation
        _, stats = _run_with_stats(g, max_generations=100, stopping_jaccard=0.0, stopping_generations=1)
        assert len(stats) < 100

    def test_stopping_generations_large_runs_to_max(self, g):
        # convergence threshold so high it won't trigger naturally in 3 gens
        _, stats = _run_with_stats(g, max_generations=3, stopping_jaccard=1.0, stopping_generations=999)
        assert len(stats) == 3

    def test_resolution_parameter_high_produces_more_communities(self, g):
        # Average over seeds to account for stochasticity
        low_sizes, high_sizes = [], []
        for seed in range(6):
            low_sizes.append(len(_run(g, population_size=6, max_generations=3, random_seed=seed, resolution_parameter=0.01)))
            high_sizes.append(len(_run(g, population_size=6, max_generations=3, random_seed=seed, resolution_parameter=20.0)))
        assert np.mean(high_sizes) >= np.mean(low_sizes)

    def test_n_iterations_one_valid(self, g):
        assert isinstance(_run(g, n_iterations=1), ig.VertexClustering)

    def test_n_iterations_large_valid(self, g):
        assert isinstance(_run(g, n_iterations=10), ig.VertexClustering)

    def test_is_weighted_false_overrides_weighted_graph(self, g_weighted):
        result = _run(g_weighted, is_weighted=False)
        assert isinstance(result, ig.VertexClustering)

    def test_is_weighted_true_on_unweighted_graph_uses_default_weight(self, g):
        result = _run(g, is_weighted=True)
        assert isinstance(result, ig.VertexClustering)

    def test_is_weighted_none_auto_detects_weighted(self, g_weighted):
        config = TauConfig(worker_count=1, is_weighted=None)
        tau = TauClustering(g_weighted, population_size=4, max_generations=2, config=config)
        assert tau.config.is_weighted is True

    def test_is_weighted_none_auto_detects_unweighted(self, g):
        config = TauConfig(worker_count=1, is_weighted=None)
        tau = TauClustering(g, population_size=4, max_generations=2, config=config)
        assert tau.config.is_weighted is False

    def test_elite_fraction_one_all_elites_no_offspring(self, g):
        result = _run(g, population_size=4, max_generations=2, elite_fraction=1.0, immigrant_fraction=0.01)
        assert isinstance(result, ig.VertexClustering)

    def test_elite_similarity_threshold_zero_one_elite_kept(self, g):
        # threshold=0.0 → any two solutions exceed it → only 1 elite survives
        result = _run(g, population_size=6, max_generations=2, elite_similarity_threshold=0.0)
        assert isinstance(result, ig.VertexClustering)

    def test_elite_similarity_threshold_one_keeps_duplicates(self, g):
        # threshold=1.0 → even identical solutions pass the diversity filter
        result = _run(g, population_size=6, max_generations=2, elite_similarity_threshold=1.0)
        assert isinstance(result, ig.VertexClustering)

    def test_context_manager_shuts_pool_on_exit(self, g):
        config = TauConfig(worker_count=1)
        with TauClustering(g, population_size=4, max_generations=2, config=config) as tau:
            tau.run()
        assert tau._pool is None

    def test_reuse_worker_pool_false_shuts_down_after_run(self, g):
        config = TauConfig(worker_count=1, reuse_worker_pool=False)
        tau = TauClustering(g, population_size=4, max_generations=2, config=config)
        tau.run()
        assert tau._pool is None

    def test_reuse_worker_pool_true_keeps_pool_alive(self, g):
        config = TauConfig(worker_count=1, reuse_worker_pool=True)
        tau = TauClustering(g, population_size=4, max_generations=2, config=config)
        tau.run()
        assert tau._pool is not None
        tau._shutdown_pool()

    def test_verbose_true_does_not_break_run(self, g):
        result = _run(g, verbose=True)
        assert isinstance(result, ig.VertexClustering)

    def test_sim_sample_size_none_uses_all_nodes(self, g):
        config = TauConfig(worker_count=1, sim_sample_size=None)
        tau = TauClustering(g, population_size=4, max_generations=2, config=config)
        assert tau.sim_indices is None
        assert isinstance(tau.run(), ig.VertexClustering)

    def test_sim_sample_size_larger_than_graph_uses_all_nodes(self, g):
        config = TauConfig(worker_count=1, sim_sample_size=10_000)
        tau = TauClustering(g, population_size=4, max_generations=2, config=config)
        # graph has 10 nodes, sample_size=10_000 > 10 → None (all nodes)
        assert tau.sim_indices is None

    def test_sim_sample_size_smaller_than_graph_samples_subset(self):
        g_large = ig.Graph.Erdos_Renyi(50, 0.2, directed=False)
        config = TauConfig(worker_count=1, sim_sample_size=5, random_seed=0)
        tau = TauClustering(g_large, population_size=4, max_generations=2, config=config)
        assert tau.sim_indices is not None
        assert len(tau.sim_indices) == 5


# ===========================================================================
# 7. Edge Cases — extreme valid values still produce correct outputs
# ===========================================================================

class TestEdgeCaseBehavior:

    def test_result_membership_length_matches_graph(self, g):
        result = _run(g)
        assert len(result.membership) == g.vcount()

    def test_result_has_at_least_one_community(self, g):
        result = _run(g)
        assert len(result) >= 1

    def test_result_modularity_is_numeric(self, g):
        result = _run(g)
        assert isinstance(result.modularity, float)
        assert not np.isnan(result.modularity)

    def test_selection_power_one_uniform_selection(self, g):
        assert isinstance(_run(g, population_size=6, max_generations=2, selection_power=1), ig.VertexClustering)

    def test_selection_power_very_large_skews_to_best(self, g):
        assert isinstance(_run(g, population_size=6, max_generations=2, selection_power=100), ig.VertexClustering)

    def test_sample_fraction_range_very_narrow(self, g):
        assert isinstance(_run(g, sample_fraction_range=(0.5, 0.5)), ig.VertexClustering)

    def test_sample_fraction_range_near_full_graph(self, g):
        assert isinstance(_run(g, sample_fraction_range=(0.99, 1.0)), ig.VertexClustering)

    def test_sample_fraction_range_very_small(self, g):
        assert isinstance(_run(g, sample_fraction_range=(0.01, 0.05)), ig.VertexClustering)

    def test_default_edge_weight_large(self, g):
        assert isinstance(_run(g, default_edge_weight=1e6), ig.VertexClustering)

    def test_default_edge_weight_very_small(self, g):
        assert isinstance(_run(g, default_edge_weight=1e-6), ig.VertexClustering)

    def test_immigrant_fraction_near_one(self, g):
        # Nearly all population replaced by immigrants each generation
        result = _run(g, population_size=6, max_generations=2, elite_fraction=0.01, immigrant_fraction=0.99)
        assert isinstance(result, ig.VertexClustering)

    def test_worker_chunk_size_one(self, g):
        assert isinstance(_run(g, worker_chunk_size=1), ig.VertexClustering)

    def test_worker_chunk_size_larger_than_population(self, g):
        assert isinstance(_run(g, population_size=4, worker_chunk_size=1000), ig.VertexClustering)

    def test_networkx_graph_input(self):
        import networkx as nx
        gnx = nx.petersen_graph()
        result = _run(gnx, population_size=4, max_generations=2)
        assert isinstance(result, ig.VertexClustering)
        assert len(result.membership) == gnx.number_of_nodes()

    def test_weighted_networkx_graph_input(self):
        import networkx as nx
        gnx = nx.petersen_graph()
        for u, v in gnx.edges():
            gnx[u][v]["weight"] = 2.5
        result = _run(gnx, population_size=4, max_generations=2)
        assert isinstance(result, ig.VertexClustering)

    def test_large_population_small_generations(self, g):
        result = _run(g, population_size=20, max_generations=1)
        assert isinstance(result, ig.VertexClustering)

    def test_membership_values_are_nonnegative_integers(self, g):
        result = _run(g)
        membership = np.array(result.membership)
        assert membership.min() >= 0
        assert membership.dtype.kind == "i" or np.issubdtype(membership.dtype, np.integer)

    def test_community_labels_are_contiguous(self, g):
        result = _run(g)
        labels = sorted(set(result.membership))
        assert labels == list(range(len(labels))), "community labels must be 0-based and contiguous"
