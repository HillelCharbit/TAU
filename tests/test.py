from pathlib import Path
import sys
import csv

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


NUM_ITERATIONS = 10
RESULTS_CSV = ROOT / "tests" / "tau_timings.csv"


def main():
    
    graph_sizes = [10000, 20000, 50000, 100000]
    results = []
    for size in graph_sizes:
        
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
            clustering = TauClustering(graph, population_size=64, max_generations=50, config=tau_config)
            _, _, generation_stats = clustering.run(track_stats=True)

            def _mean_std(stats_list: list[dict[str, float]], key: str) -> tuple[float, float]:
                values = [
                    float(entry[key])
                    for entry in stats_list
                    if isinstance(entry, dict) and entry.get(key) is not None
                ]
                if not values:
                    return float("nan"), float("nan")
                mean_val = float(np.mean(values))
                std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                return mean_val, std_val

            mean_time, std_time = _mean_std(generation_stats, "time_per_generation")
            mean_elt, std_elt = _mean_std(generation_stats, "elite_runtime")
            mean_crim, std_crim = _mean_std(generation_stats, "crossover_runtime")
            
            results.append(
                {
                    "graph_size": size,
                    "iteration": iteration,
                    "mean_time_per_generation": mean_time,
                    "std_time_per_generation": std_time,
                    "mean_elite_runtime": mean_elt,
                    "std_elite_runtime": std_elt,
                    "mean_crossover_runtime": mean_crim,
                    "std_crossover_runtime": std_crim,
                }
            )
    if results:
        with RESULTS_CSV.open("w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "graph_size",
                    "iteration",
                    "mean_time_per_generation",
                    "std_time_per_generation",
                    "mean_elite_runtime",
                    "std_elite_runtime",
                    "mean_crossover_runtime",
                    "std_crossover_runtime",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved timing data to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
