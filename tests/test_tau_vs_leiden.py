import os
import sys
import time
import argparse
import igraph as ig
import numpy as np
import cProfile
import pstats
import io

# Use local tau_community_detection before installed library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from tau_bottleneck import load_graph_file, get_subgraph_bfs
import tau_community_detection.algorithm as algorithm
from tau_community_detection import TauClustering, TauConfig

class ProfilingSequentialPool(algorithm._SequentialPool):
    def __init__(self, *args, **kwargs):
        pass

def measure_leiden_avg_time(g: ig.Graph, runs: int = 5) -> float:
    print(f"Running Leiden algorithm {runs} times to calculate average...")
    weights = g.es["weight"] if "weight" in g.es.attributes() else None
    
    times = []
    for i in range(runs):
        start = time.time()
        partition = g.community_leiden(objective_function="modularity", weights=weights, n_iterations=3)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Leiden run {i+1}: {elapsed:.4f} seconds")
        print(f" modularity score: {partition.modularity}  ")
        
    avg_time = sum(times) / len(times)
    return avg_time

def measure_tau_generation_avg_time(g: ig.Graph, max_generations: int, pop_size: int, worker_count: int = None) -> float:
    print(f"\nRunning TAU Clustering (generations={max_generations}, pop_size={pop_size})...")
    
    workers = worker_count if worker_count is not None else (os.cpu_count() or 4)
    config = TauConfig(
        population_size=pop_size,
        worker_count=workers,
        max_generations=max_generations,
        is_weighted=True,
        verbose=True
    )
    
    tau = TauClustering(
        g,
        population_size=pop_size,
        max_generations=max_generations,
        config=config
    )
    
    start_total = time.time()
    result = tau.run(track_stats=True)
    total_time = time.time() - start_total
    
    if isinstance(result, tuple) and len(result) == 2:
        clustering_result, generation_stats = result
    else:
        print("  Error: track_stats=True did not return generation_stats.")
        return 0.0, 0, 0, 0.0, 0, 0
        
    if not generation_stats:
        print("  No generation stats collected.")
        return 0.0, 0, 0, 0.0, 0, 0
        
    gen_times = [stat["time_per_generation"] for stat in generation_stats]
    avg_gen_time = sum(gen_times) / len(gen_times)
    
    # Calculate Leiden calls based on algorithm logic
    immigrant_count = max(1, int(config.immigrant_fraction * config.population_size))
    init_pop_calls = config.population_size
    
    # TAU performs an additional optimization step before breaking
    actual_generations = len(gen_times)
    
    # 1 init batch + (K+1) optimization batches (immigrants are run concurrently with optimization now!)
    sequential_batches = 1 + (actual_generations + 1)
    
    # +1 ghost immigrant call generated eagerly before the break!
    total_leiden_calls = init_pop_calls + (actual_generations + 1) * config.population_size + actual_generations * immigrant_count + immigrant_count
    
    print(f"  TAU ran for {actual_generations} generations in {total_time:.4f}s total.")
    return avg_gen_time, init_pop_calls, immigrant_count, total_time, actual_generations, total_leiden_calls, sequential_batches

def profile_tau_leiden_calls(pop_size: int, max_generations: int):
    print(f"\n--- Profiling Leiden Calls (1 worker, sequential) ---")
    print("Running TAU on a tiny dummy graph just to count Leiden calls...")
    g_tiny = ig.Graph.Erdos_Renyi(100, 0.1)
    g_tiny.es["weight"] = np.random.rand(g_tiny.ecount())
    
    config = TauConfig(
        population_size=pop_size,
        worker_count=1,
        max_generations=max_generations,
        is_weighted=True,
        verbose=False
    )
    
    tau = TauClustering(g_tiny, population_size=pop_size, max_generations=max_generations, config=config)
    
    original_pool = algorithm.Pool
    algorithm.Pool = ProfilingSequentialPool
    
    pr = cProfile.Profile()
    pr.enable()
    tau.run(track_stats=False)
    pr.disable()
    
    algorithm.Pool = original_pool
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('calls')
    ps.print_stats('community_leiden')
    
    print(s.getvalue())
    print("-----------------------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Compare TAU generation time vs Leiden call time")
    parser.add_argument("--graph", type=str, default="proofread_filtered.graph", help="Path to the graph file (default: proofread_filtered.graph)")
    parser.add_argument("--size", type=int, default=0, help="Graph size to test (uses BFS subgraph). Set to 0 to use the full graph.")
    parser.add_argument("--leiden_runs", type=int, default=5, help="Number of Leiden runs for averaging")
    parser.add_argument("--generations", type=int, default=5, help="Max TAU generations to run")
    parser.add_argument("--pop_size", type=int, default=10, help="Population size for TAU")
    parser.add_argument("--worker_count", type=int, default=None, help="Number of workers for TAU (default: use all cores)")
    args = parser.parse_args()

    if not os.path.exists(args.graph):
        print(f"Error: Graph file '{args.graph}' not found.")
        print(f"Please run from the directory containing the file, or provide --graph.")
        return
    # Run standalone fast profiling for Leiden call count verification
    profile_tau_leiden_calls(args.pop_size, args.generations)
    
    g = load_graph_file(args.graph)
    
    if args.size and args.size > 0 and args.size < g.vcount():
        g = get_subgraph_bfs(g, args.size)
        
    print(f"\n--- Testing on Graph with {g.vcount()} nodes, {g.ecount()} edges ---")
    
    avg_leiden_time = measure_leiden_avg_time(g, runs=args.leiden_runs)
    avg_tau_gen_time, init_pop_calls, immigrant_count, total_tau_time, actual_generations, total_leiden_calls, sequential_batches = measure_tau_generation_avg_time(g, args.generations, args.pop_size, args.worker_count)
    
    print("\n" + "="*80)
    print("--- Summary ---")
    print(f"Average time of a single Leiden call:  {avg_leiden_time:.4f} seconds")
    print(f"Total TAU run time:                    {total_tau_time:.4f} seconds")
    print(f"Number of recorded generations:        {actual_generations}")
    print(f"Average time per TAU generation:       {avg_tau_gen_time:.4f} seconds")
    
    print(f"\nLeiden Call Breakdown:")
    print(f"  - Initial Population: {init_pop_calls} calls (on subgraphs)")
    print(f"  - Optimization steps: {(actual_generations + 1) * args.pop_size} calls ({actual_generations + 1} steps * {args.pop_size} calls on full graph)")
    print(f"  - Immigrant creation: {actual_generations * immigrant_count} calls (concurrent with optimizations)")
    print(f"  - Eager Immigrant discarded at break: {immigrant_count} calls (1 ghost batch)")
    print(f"  - Total Leiden calls: {total_leiden_calls} calls distributed across ONLY {sequential_batches} sequential batches!")
    
    if avg_leiden_time > 0:
        expected_parallel_time = sequential_batches * avg_leiden_time
        ratio = total_tau_time / expected_parallel_time
        print(f"\nApples-to-Apples Parallel Overhead:")
        print(f"  Expected time for {sequential_batches} sequential batches: ~{expected_parallel_time:.2f}s")
        print(f"  Actual TAU parallel time:                    {total_tau_time:.2f}s")
        print(f"  Ratio (TAU total time / Expected sequential batches time): {ratio:.2f}x")
    print("="*80 + "\n")
    

if __name__ == "__main__":
    main()
