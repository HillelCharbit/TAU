"""Tests for graph loading functionality."""
from pathlib import Path

import igraph as ig
import networkx as nx
import pytest

from tau_community_detection.graph import (
    load_graph,
    load_graph_worker,
    _is_adjlist_format,
    _detect_weighted,
)


# =============================================================================
# NetworkX Input Tests
# =============================================================================

class TestNetworkXInput:
    """Tests for loading from NetworkX graphs."""

    def test_unweighted_karate(self):
        """Note: karate_club_graph actually HAS weights (edge strength)."""
        G = nx.karate_club_graph()
        graph, weighted, path = load_graph(G)
        
        assert isinstance(graph, ig.Graph)
        assert graph.vcount() == 34
        assert graph.ecount() == 78
        assert weighted  # karate club has weights!
        assert Path(path).exists()

    def test_truly_unweighted_graph(self):
        """Test with a graph that has no edge attributes."""
        G = nx.cycle_graph(10)
        graph, weighted, path = load_graph(G)
        
        assert not weighted
        assert all(w == 1.0 for w in graph.es["weight"])

    def test_weighted_simple(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=0.5)
        G.add_edge(1, 2, weight=1.5)
        G.add_edge(2, 0, weight=2.0)
        
        graph, weighted, path = load_graph(G)
        
        assert weighted
        assert graph.ecount() == 3
        assert set(graph.es["weight"]) == {0.5, 1.5, 2.0}

    def test_custom_weight_attr(self):
        G = nx.Graph()
        G.add_edge(0, 1, strength=0.5)
        G.add_edge(1, 2, strength=1.5)
        
        graph, weighted, path = load_graph(G, weight_attr="strength")
        
        assert weighted

    def test_force_unweighted_ignores_weights(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=0.5)
        G.add_edge(1, 2, weight=1.5)
        
        graph, weighted, path = load_graph(G, is_weighted=False)
        
        assert not weighted
        assert all(w == 1.0 for w in graph.es["weight"])

    def test_force_weighted_uses_default(self):
        """Force weighted on an unweighted graph - should use default."""
        G = nx.cycle_graph(5)  # no weights
        
        graph, weighted, path = load_graph(G, is_weighted=True, default_weight=0.5)
        
        assert weighted
        assert all(w == 0.5 for w in graph.es["weight"])

    def test_directed_preserved(self):
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        
        graph, _, _ = load_graph(G)
        
        assert graph.is_directed()

    def test_string_node_names(self):
        G = nx.Graph()
        G.add_edge("alice", "bob", weight=1.0)
        G.add_edge("bob", "charlie", weight=2.0)
        
        graph, weighted, path = load_graph(G)
        
        assert graph.vcount() == 3
        assert graph.ecount() == 2
        assert weighted


# =============================================================================
# igraph Input Tests
# =============================================================================

class TestIgraphInput:
    """Tests for loading from igraph graphs."""

    def test_unweighted_famous(self):
        G = ig.Graph.Famous("Zachary")
        graph, weighted, path = load_graph(G)
        
        assert graph.vcount() == 34
        assert graph.ecount() == 78
        assert not weighted  # igraph Famous doesn't add weights
        assert Path(path).exists()

    def test_weighted(self):
        G = ig.Graph.Famous("Zachary")
        G.es["weight"] = [0.5] * G.ecount()
        
        graph, weighted, path = load_graph(G)
        
        assert weighted
        assert all(w == 0.5 for w in graph.es["weight"])

    def test_original_not_modified(self):
        G = ig.Graph.Famous("Zachary")
        original_attrs = set(G.es.attributes())
        
        load_graph(G)
        
        assert set(G.es.attributes()) == original_attrs

    def test_force_unweighted_on_weighted(self):
        G = ig.Graph.Ring(10)
        G.es["weight"] = [i * 0.1 for i in range(G.ecount())]
        
        graph, weighted, path = load_graph(G, is_weighted=False)
        
        assert not weighted
        assert all(w == 1.0 for w in graph.es["weight"])


# =============================================================================
# Edgelist File Tests
# =============================================================================

class TestEdgelistFile:
    """Tests for loading from edgelist/NCOL files."""

    def test_unweighted(self, tmp_path):
        edgelist = tmp_path / "graph.edgelist"
        edgelist.write_text("0 1\n1 2\n2 0\n")
        
        graph, weighted, path = load_graph(str(edgelist))
        
        assert graph.vcount() == 3
        assert graph.ecount() == 3
        assert not weighted
        assert path == str(edgelist)  # returns original path, not temp

    def test_weighted(self, tmp_path):
        edgelist = tmp_path / "graph.edgelist"
        edgelist.write_text("0 1 0.5\n1 2 1.5\n2 0 2.0\n")
        
        graph, weighted, path = load_graph(str(edgelist))
        
        assert weighted
        assert set(graph.es["weight"]) == {0.5, 1.5, 2.0}

    def test_comments_ignored(self, tmp_path):
        edgelist = tmp_path / "graph.edgelist"
        edgelist.write_text("# header comment\n0 1\n1 2\n# middle comment\n2 0\n")
        
        graph, weighted, path = load_graph(str(edgelist))
        
        assert graph.vcount() == 3
        assert graph.ecount() == 3

    def test_force_weighted_applies_default(self, tmp_path):
        edgelist = tmp_path / "graph.edgelist"
        edgelist.write_text("0 1\n1 2\n2 0\n")
        
        graph, weighted, path = load_graph(str(edgelist), is_weighted=True, default_weight=0.7)
        
        assert weighted
        assert all(w == 0.7 for w in graph.es["weight"])

    def test_named_nodes(self, tmp_path):
        edgelist = tmp_path / "graph.ncol"
        edgelist.write_text("alice bob 1.0\nbob charlie 2.0\n")
        
        graph, weighted, path = load_graph(str(edgelist))
        
        assert graph.vcount() == 3
        assert weighted


# =============================================================================
# Adjacency List File Tests
# =============================================================================

class TestAdjlistFile:
    """Tests for loading from adjacency list files."""

    def test_simple(self, tmp_path):
        # Must have >3 columns to be detected as adjlist
        adjlist = tmp_path / "graph.adjlist"
        adjlist.write_text("0 1 2 3\n1 0 2\n2 0 1\n3 0\n")
        
        graph, weighted, path = load_graph(str(adjlist))
        
        assert graph.vcount() == 4
        assert not weighted

    def test_lfr_style(self, tmp_path):
        adjlist = tmp_path / "network.dat"
        adjlist.write_text("1 2 3 4 5\n2 1 3\n3 1 2\n4 1\n5 1\n")
        
        graph, weighted, path = load_graph(str(adjlist))
        
        assert graph.vcount() == 5

    def test_with_comments(self, tmp_path):
        adjlist = tmp_path / "graph.adjlist"
        adjlist.write_text("# nodes and neighbors\n0 1 2 3 4\n1 0 2\n")
        
        graph, _, _ = load_graph(str(adjlist))
        
        assert graph.vcount() >= 2


# =============================================================================
# Format Detection Tests
# =============================================================================

class TestFormatDetection:
    """Tests for internal format detection functions."""

    def test_is_adjlist_true(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("0 1 2 3 4\n")  # 5 columns -> adjlist
        
        assert _is_adjlist_format(str(f))

    def test_is_adjlist_false_edgelist(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("0 1\n")  # 2 columns -> edgelist
        
        assert not _is_adjlist_format(str(f))

    def test_is_adjlist_false_weighted(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("0 1 0.5\n")  # 3 columns -> weighted edgelist
        
        assert not _is_adjlist_format(str(f))

    def test_is_adjlist_skips_comments(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("# comment\n0 1 2 3 4\n")
        
        assert _is_adjlist_format(str(f))

    def test_detect_weighted_true(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("0 1 0.5\n")
        
        assert _detect_weighted(str(f))

    def test_detect_weighted_false(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("0 1\n")
        
        assert not _detect_weighted(str(f))

    def test_detect_weighted_non_numeric_third_col(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("alice bob friend\n")  # third col not numeric
        
        assert not _detect_weighted(str(f))


# =============================================================================
# Worker Loading Tests
# =============================================================================

class TestWorkerLoading:
    """Tests for worker-side graph loading."""

    def test_unweighted(self, tmp_path):
        f = tmp_path / "graph.ncol"
        f.write_text("0 1\n1 2\n2 0\n")
        
        graph = load_graph_worker(str(f), is_weighted=False)
        
        assert graph.vcount() == 3
        assert all(w == 1.0 for w in graph.es["weight"])

    def test_weighted(self, tmp_path):
        f = tmp_path / "graph.ncol"
        f.write_text("0 1 0.5\n1 2 1.5\n2 0 2.0\n")
        
        graph = load_graph_worker(str(f), is_weighted=True)
        
        assert set(graph.es["weight"]) == {0.5, 1.5, 2.0}

    def test_custom_default_weight(self, tmp_path):
        f = tmp_path / "graph.ncol"
        f.write_text("0 1\n1 2\n")
        
        graph = load_graph_worker(str(f), default_weight=0.3, is_weighted=False)
        
        assert all(w == 0.3 for w in graph.es["weight"])


# =============================================================================
# Roundtrip / Integration Tests
# =============================================================================

class TestRoundtrip:
    """Tests for full roundtrip scenarios."""

    def test_nx_to_temp_to_worker(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=0.5)
        G.add_edge(1, 2, weight=1.5)
        
        graph, weighted, path = load_graph(G)
        worker_graph = load_graph_worker(path, is_weighted=weighted)
        
        assert worker_graph.vcount() == graph.vcount()
        assert worker_graph.ecount() == graph.ecount()
        assert set(worker_graph.es["weight"]) == set(graph.es["weight"])

    def test_igraph_to_temp_to_worker(self):
        G = ig.Graph.Ring(10)
        G.es["weight"] = [i * 0.1 for i in range(G.ecount())]
        
        graph, weighted, path = load_graph(G)
        worker_graph = load_graph_worker(path, is_weighted=weighted)
        
        assert worker_graph.vcount() == graph.vcount()
        assert worker_graph.ecount() == graph.ecount()

    def test_temp_file_valid_ncol(self):
        G = nx.cycle_graph(10)  # truly unweighted
        graph, weighted, path = load_graph(G)
        
        # Verify temp file is valid NCOL
        reloaded = ig.Graph.Read_Ncol(path, weights=False, directed=False)
        assert reloaded.vcount() == graph.vcount()
        assert reloaded.ecount() == graph.ecount()

    def test_file_path_not_copied(self, tmp_path):
        """File paths should return original path, not temp file."""
        edgelist = tmp_path / "graph.edgelist"
        edgelist.write_text("0 1\n1 2\n")
        
        _, _, path = load_graph(str(edgelist))
        
        assert path == str(edgelist)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_graph_nx(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        
        graph, weighted, path = load_graph(G)
        
        assert graph.vcount() == 3
        assert graph.ecount() == 0

    def test_empty_graph_igraph(self):
        G = ig.Graph(n=5)
        
        graph, weighted, path = load_graph(G)
        
        assert graph.vcount() == 5
        assert graph.ecount() == 0

    def test_self_loops_nx(self):
        G = nx.Graph()
        G.add_edge(0, 0)
        G.add_edge(0, 1)
        
        graph, _, _ = load_graph(G)
        
        assert graph.ecount() == 2

    def test_mixed_weight_attrs(self):
        """Some edges have weight, some don't."""
        G = nx.Graph()
        G.add_edge(0, 1, weight=0.5)
        G.add_edge(1, 2)  # no weight
        
        graph, weighted, path = load_graph(G)
        
        assert weighted
        assert graph.ecount() == 2

    def test_large_graph(self):
        """Verify handling of larger graphs."""
        G = nx.barabasi_albert_graph(1000, 5)
        
        graph, weighted, path = load_graph(G)
        
        assert graph.vcount() == 1000
        assert Path(path).exists()