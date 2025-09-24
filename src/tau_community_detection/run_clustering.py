"""Example script demonstrating programmatic use of TAU clustering."""
from __future__ import annotations

import sys
from pathlib import Path


# Ensure the local ``tau_community_detection`` package is importable when the
# script is executed via ``python src/tau_community_detection/run_clustering.py``.
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tau_community_detection import TauClustering
from tau_community_detection.config import TauConfig


if __name__ == "__main__":
    import networkx as nx
    from tempfile import NamedTemporaryFile

    # INSERT_YOUR_CODE
    # Generate a random Erdos-Renyi graph with 1000 vertices and probability p=0.01
    
    G = nx.erdos_renyi_graph(7000, 0.005)

    print("=== Unweighted run (NetworkX graph) ===")
    unweighted_config = TauConfig(leiden_resolution=1.1, weight_attribute=None)
    clustering_unweighted = TauClustering(G, population_size=40, max_generation=50, config=unweighted_config)
    membership_unweighted, history_unweighted = clustering_unweighted.run()
    print(f"Unweighted communities: {len(set(membership_unweighted))}")
    print(f"Unweighted final modularity: {history_unweighted[-1] if history_unweighted else 'n/a'}")

    print("\n=== Weighted run (NetworkX graph) ===")
    weighted_graph = G.copy()
    for u, v in weighted_graph.edges():
        weighted_graph[u][v]["weight"] = 1.0 + abs(u - v) / len(weighted_graph)

    weighted_config = TauConfig(leiden_resolution=1.1, weight_attribute="weight")
    clustering_weighted = TauClustering(weighted_graph, population_size=40, max_generation=50, config=weighted_config)
    membership_weighted, history_weighted = clustering_weighted.run()
    print(f"Weighted communities: {len(set(membership_weighted))}")
    print(f"Weighted final modularity: {history_weighted[-1] if history_weighted else 'n/a'}")

    # unweighted_path = None
    # weighted_path = None
    # try:
    #     print("\n=== Unweighted run (graph path) ===")
    #     with NamedTemporaryFile(delete=False, suffix=".adjlist") as fh:
    #         unweighted_path = Path(fh.name)
    #     nx.write_adjlist(G, unweighted_path)
    #     clustering_from_path = TauClustering(unweighted_path, population_size=40, max_generation=100, config=unweighted_config)
    #     membership_path, history_path = clustering_from_path.run()
    #     print(f"Unweighted (path) communities: {len(set(membership_path))}")
    #     print(f"Unweighted (path) final modularity: {history_path[-1] if history_path else 'n/a'}")

    #     print("\n=== Weighted run (graph path) ===")
    #     with NamedTemporaryFile(delete=False, suffix=".edgelist") as fh:
    #         weighted_path = Path(fh.name)
    #     nx.write_weighted_edgelist(weighted_graph, weighted_path)
    #     clustering_weighted_path = TauClustering(weighted_path, population_size=40, max_generation=100, config=weighted_config)
    #     membership_weighted_path, history_weighted_path = clustering_weighted_path.run()
    #     print(f"Weighted (path) communities: {len(set(membership_weighted_path))}")
    #     print(f"Weighted (path) final modularity: {history_weighted_path[-1] if history_weighted_path else 'n/a'}")
    # finally:
    #     for maybe_path in (unweighted_path, weighted_path):
    #         if maybe_path is not None and maybe_path.exists():
    #             maybe_path.unlink()
