from pathlib import Path
import sys

# Ensure the in-repo package is importable when running the script directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    src_str = str(SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

import networkx as nx
import igraph as ig
from tau_community_detection import TauClustering, TauConfig
 
import numpy as np
def main():
    # seed = 42
    # graph = nx.LFR_benchmark_graph(n=10000, tau1=3, tau2=1.5, mu=0.3, average_degree=15, min_community=20,
    #                                seed=seed, max_iters=1_000, max_degree=60)

    # print(f"Generated LFR graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    # Read the 200k node graph in adjacency list format into a NetworkX graph
    # INSERT_YOUR_CODE
    graph_sizes = ["100k", "200k", "500k", "1M"]
    for size in graph_sizes:
        graph_path = f"tests/test_instance-{size}.graph"
        print(f"\n=== Running TAU on {size} graph ===")
        graph = nx.read_adjlist(graph_path, nodetype=int)
        print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges from {graph_path}")

        clustering = TauClustering(graph, population_size=64, max_generations=50)
        mod_history, total_time, elt_time, crim_time = clustering.run()

        print("Best modularity:", mod_history[-1])
        print("time per generation:", np.mean(total_time))
        print("elite time per generation:", np.mean(elt_time))
        print("crim time per generation:", np.mean(crim_time))

        # ig_graph = ig.Graph.from_networkx(graph)
        # leiden_membership = ig_graph.community_leiden(
        #     objective_function="modularity",
        #     n_iterations=-1,
        #     resolution_parameter=1.0,
        #     weights=None,
        # )
        # modularity = ig_graph.modularity(leiden_membership)
        # print(f"Leiden modularity: {modularity}")
    
if __name__ == "__main__":
    main()
