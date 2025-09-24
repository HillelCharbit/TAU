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


if __name__ == "__main__":
    import networkx as nx
    
    # INSERT_YOUR_CODE
    # Generate a random Erdos-Renyi graph with 1000 vertices and probability p=0.01
    
    G = nx.erdos_renyi_graph(1000, 0.1)
    graph_path = PACKAGE_DIR / "examples" / "example.graph"
    clustering = TauClustering(G, population_size=60, max_generation=150)
    membership, mod_history = clustering.run()

    # print(membership)
    # print(mod_history)
