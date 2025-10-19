from pathlib import Path
import sys
import csv
import time

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


NUM_ITERATIONS = 3
RESULTS_CSV = ROOT / "tests" / "tau_timings.csv"


def main():
    # seed = 42
    # graph = nx.LFR_benchmark_graph(n=10000, tau1=3, tau2=1.5, mu=0.3, average_degree=15, min_community=20,
    #                                seed=seed, max_iters=1_000, max_degree=60)

    # print(f"Generated LFR graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    # Read the 200k node graph in adjacency list format into a NetworkX graph
    # INSERT_YOUR_CODE
    graph_sizes = [10000, 20000, 50000, 100000]
    results = []
    for size in graph_sizes:
        # graph_path = ROOT / "tests" / f"test_instance-{size}.graph"
        # generate_graph_if_needed(size, graph_path)

        # print(f"\n=== Running TAU on {size} graph ===")
        # graph = nx.read_adjlist(graph_path, nodetype=int)
        # print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges from {graph_path}")
        

        last_mod_history = None
        last_total_time = None
        last_elt_time = None
        last_crim_time = None
        run_times = []
        tau_config = TauConfig(population_size=64, max_generations=50, stopping_generations=50)
        for iteration in range(1, NUM_ITERATIONS + 1):
            graph = nx.LFR_benchmark_graph(
                n=size,
                tau1=3,
                tau2=1.5,
                mu=0.3,
                average_degree=15,
                min_community=20,
                max_degree=60,
                max_iters=1_000,
                seed=42,
            )
            print(f"Generated LFR graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
            start_time = time.perf_counter()
            clustering = TauClustering(graph, population_size=64, max_generations=50, config=tau_config)
            run_result = clustering.run()
            if isinstance(run_result, tuple) and len(run_result) == 4:
                mod_history, total_time, elt_time, crim_time = run_result
            elif isinstance(run_result, tuple) and len(run_result) == 2:
                _, mod_history = run_result
                total_time = elt_time = crim_time = None
            else:
                mod_history = list(run_result) if run_result else []
                total_time = elt_time = crim_time = None
            elapsed = time.perf_counter() - start_time
            run_times.append(elapsed)

            last_mod_history = mod_history
            last_total_time = total_time
            last_elt_time = elt_time
            last_crim_time = crim_time

            results.append(
                {
                    "graph_size": size,
                    "iteration": iteration,
                    "elapsed_seconds": elapsed,
                }
            )

        if last_mod_history is not None:
            print("Best modularity:", last_mod_history[-1])
        if last_total_time is not None:
            print("time per generation:", np.mean(last_total_time))
        if last_elt_time is not None:
            print("elite time per generation:", np.mean(last_elt_time))
        if last_crim_time is not None:
            print("crim time per generation:", np.mean(last_crim_time))

        avg_time = np.mean(run_times)
        std_time = np.std(run_times, ddof=1) if len(run_times) > 1 else 0.0
        print(f"Runtime over {NUM_ITERATIONS} iteration(s): mean={avg_time:.2f}s std={std_time:.2f}s")

        # ig_graph = ig.Graph.from_networkx(graph)
        # leiden_membership = ig_graph.community_leiden(
        #     objective_function="modularity",
        #     n_iterations=-1,
        #     resolution_parameter=1.0,
        #     weights=None,
        # )
        # modularity = ig_graph.modularity(leiden_membership)
        # print(f"Leiden modularity: {modularity}")

    if results:
        with RESULTS_CSV.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["graph_size", "iteration", "elapsed_seconds"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved timing data to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
