from pathlib import Path
from typing import Optional
import sys
import numpy as np

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

def _build_weighted_lfr_graph(
    seed: int = 123,
    node_count: int = 20_000,
    tau1: float = 2.5,
    tau2: float = 1.5,
    mu: float = 0.55,
    average_degree: Optional[int] = 30,
    min_degree: Optional[int] = None,
    max_degree: int = 80,
    min_community: int = 80,
    max_community: int = 600,
) -> tuple[nx.Graph, np.ndarray]:
    """Generate a large weighted LFR benchmark graph."""

    degree_params = {
        "average_degree": average_degree,
        "min_degree": min_degree,
    }
    provided_degree_params = {k: v for k, v in degree_params.items() if v is not None}
    if len(provided_degree_params) != 1:
        raise ValueError("Must specify exactly one of average_degree or min_degree.")

    graph = nx.LFR_benchmark_graph(
        n=node_count,
        tau1=tau1,
        tau2=tau2,
        mu=mu,
        max_degree=max_degree,
        min_community=min_community,
        max_community=max_community,
        seed=seed,
        **provided_degree_params,
    )

    # Assign synthetic weights: intra-community edges receive higher weights
    # than inter-community edges to mimic strength-based clustering.
    rng = np.random.default_rng(seed + 1)
    communities = {node: frozenset(graph.nodes[node]["community"]) for node in graph}

    for u, v in graph.edges():
        if communities[u] & communities[v]:
            weight = rng.normal(loc=1.6, scale=0.25)
        else:
            weight = rng.normal(loc=1.0, scale=0.35)
        weight = float(np.clip(weight, 0.05, 3.0))
        graph[u][v]["weight"] = weight * float(rng.uniform(0.8, 1.2))

    # Extract primary community label per node for evaluation.
    labels = []
    for node in graph:
        comms = sorted(communities[node])
        labels.append(int(comms[0]) if comms else -1)

    return graph, np.asarray(labels, dtype=np.int64)


def main():
    seed = 42
    graph, ground_truth = _build_weighted_lfr_graph(node_count=8000, seed=seed)
    # # Remove all "weight" edge attributes from the graph
    # for u, v, data in graph.edges(data=True):
    #     if "weight" in data:
    #         del data["weight"]

    print(f"Generated LFR graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    
    config = TauConfig(
        leiden_resolution=1.4,
        leiden_iterations=1,
        random_seed=seed,
    )
    clustering = TauClustering(graph, population_size=64, max_generations=50, config=config)
    # membership, modularity_history = clustering.run()
    # membership, modularity_history, generation_stats = clustering.run(track_stats=True)

    ig_graph = ig.Graph.from_networkx(graph)
    # Save the generated graph as a .graph instance (adjacency list format)
    nx.write_adjlist(graph, "tests/test_instance.graph")
    print("Graph saved as tests/test_instance.graph")

    resolution = config.leiden_resolution
    leiden_membership = ig_graph.community_leiden(
        objective_function="modularity",
        n_iterations=-1,
        resolution_parameter=resolution,
        weights="weight"
    )
    modularity = ig_graph.modularity(leiden_membership, weights="weight")
    print(f"Leiden modularity: {modularity}")

if __name__ == "__main__":
    main()
