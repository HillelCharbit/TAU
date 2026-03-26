import os
import time
import csv
import cProfile
import pstats
import argparse
import random
import pandas as pd
import igraph as ig
import sys
import os

# Use local package instead of pip-installed package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from tau_community_detection import TauClustering, TauConfig

# --- Configuration ---
DEFAULT_GRAPH_FILE = "proofread_filtered.graph"
RESULTS_CSV = "tau_time_complexity.csv"
PROFILE_FILE = "tau_bottleneck.prof"


def load_graph_file(graph_path: str):
    """Load .graph edge list and return igraph Graph."""
    print(f"Loading graph {graph_path}...")
    # Follow compare_connectome.py reading pattern
    edges_df = pd.read_csv(graph_path, sep=r"\s+", header=None, names=["source", "target", "weight"])
    if edges_df.empty:
        raise ValueError(f"{graph_path} is empty.")
    
    max_node = int(max(edges_df["source"].max(), edges_df["target"].max()))
    num_nodes = max_node + 1
    
    g = ig.Graph(n=num_nodes, directed=False)
    g.add_edges(list(zip(edges_df["source"], edges_df["target"])))
    g.es["weight"] = edges_df["weight"].tolist()
    
    print(f"Graph loaded with {g.vcount()} nodes and {g.ecount()} edges.")
    return g


def get_subgraph_bfs(g: ig.Graph, size: int):
    """Generate a cohesive subgraph using BFS to maintain community structure."""
    if size >= g.vcount():
        return g
    
    # Pick a random starting node with at least some degree
    degrees = g.degree()
    valid_starts = [v for v, d in enumerate(degrees) if d > 0]
    start_node = random.choice(valid_starts) if valid_starts else 0

    print(f"  Generating BFS subgraph of target size {size} from node {start_node}...")
    bfs_nodes = g.bfs(start_node)[0]
    
    # Filter unreachable nodes (represented as -1 or just past connected component)
    bfs_nodes = [n for n in bfs_nodes if n != -1]
    
    # If BFS didn't reach enough nodes (disconnected graph), add random nodes
    if len(bfs_nodes) < size:
        remaining = set(range(g.vcount())) - set(bfs_nodes)
        bfs_nodes.extend(random.sample(list(remaining), size - len(bfs_nodes)))
        
    bfs_nodes = bfs_nodes[:size]
    subgraph = g.subgraph(bfs_nodes)
    print(f"  Generated subgraph with {subgraph.vcount()} nodes and {subgraph.ecount()} edges.")
    return subgraph


import tau_community_detection.algorithm as algorithm

class MockSequentialPool:
    def __init__(self, processes=None, initializer=None, initargs=()):
        if initializer:
            initializer(*initargs)
            
    def map(self, func, iterable, chunksize=None):
        return [func(item) for item in iterable]
        
    def close(self):
        pass
        
    def join(self):
        pass

def run_tau(g: ig.Graph, max_generations: int = 50, pop_size: int = 40, worker_count: int = None):
    """Helper to initialize and run TAU clustering."""
    workers = worker_count if worker_count is not None else (os.cpu_count() or 4)
    config = TauConfig(
        population_size=pop_size,
        worker_count=workers,
        max_generations=max_generations,
        is_weighted=True,
        verbose=True,
        n_iterations=1,
        sample_fraction_range=(0.05, 0.3)
    )
    
    original_pool = algorithm.Pool
    if workers <= 1:
        algorithm.Pool = MockSequentialPool
        
    try:
        tau = TauClustering(
            g,
            population_size=pop_size,
            max_generations=max_generations,
            config=config
        )
        
        tau.run(track_stats=False)
    finally:
        algorithm.Pool = original_pool


def run_scaling_benchmark(g: ig.Graph, sizes: list, max_generations: int, pop_size: int, worker_count: int = None):
    """Run TAU on graph subsets of increasing size to test time scaling."""
    print("\n--- Starting Time Complexity Scaling Benchmark ---")
    results = []
    
    for size in sizes:
        subgraph = get_subgraph_bfs(g, size)
        if subgraph.ecount() == 0:
            print(f"  Skipping size {size}: Subgraph has 0 edges.")
            continue
            
        print(f"  Running TAU on subgraph of size {size} (Nodes: {subgraph.vcount()}, Edges: {subgraph.ecount()})...")
        
        start_time = time.time()
        run_tau(subgraph, max_generations, pop_size, worker_count)
        elapsed_time = time.time() - start_time
        
        print(f"  Execution time: {elapsed_time:.4f} seconds")
        results.append({
            'target_size': size,
            'actual_nodes': subgraph.vcount(),
            'actual_edges': subgraph.ecount(),
            'time_seconds': elapsed_time
        })
        
    # Save to CSV
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['target_size', 'actual_nodes', 'actual_edges', 'time_seconds'])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nScaling benchmark complete. Results saved to {RESULTS_CSV}")


def run_profiling(g: ig.Graph, max_generations: int, pop_size: int, profile_size: int = 5000, worker_count: int = None):
    """Run cProfile on a specific subset size and print top bottlenecks."""
    print(f"\n--- Starting cProfile Bottleneck Analysis ---")
    
    subgraph = g
    print(f"  Profiling TAU on full graph with {subgraph.vcount()} nodes and {subgraph.ecount()} edges...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    run_tau(subgraph, max_generations, pop_size, worker_count)
    profiler.disable()
    
    # Save profiler stats
    profiler.dump_stats(PROFILE_FILE)
    print(f"  Profile data saved to {PROFILE_FILE}")
    
    # Print top bottlenecks
    with open("tau_bottleneck_report.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats(30)  # write top 30 to file
        
    print("\nTop 15 Functions by Cumulative Time:")
    stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(15)
    
    print("\nTop 15 Functions by Internal Time (Per-call/total inline execution):")
    stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.TIME)
    stats.print_stats(15)


def main():
    parser = argparse.ArgumentParser(description="TAU Algorithm Time Complexity and Bottleneck Profiler")
    parser.add_argument("--graph", type=str, default=DEFAULT_GRAPH_FILE, help="Path to the graph file")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 2500, 5000, 10000, 20000], help="Sizes of subgraphs to test for scaling")
    parser.add_argument("--generations", type=int, default=50, help="Max TAU generations for scaling tests")
    parser.add_argument("--pop_size", type=int, default=40, help="Population size for TAU")
    parser.add_argument("--profile_size", type=int, default=5000, help="Size of subgraph for the full cProfile run")
    parser.add_argument("--worker_count", type=int, default=None, help="Number of workers (default is cpu_count). Set to 1 for true profiling.")
    parser.add_argument("--skip_scaling", action="store_true", help="Skip scaling benchmark, run profiling only")
    parser.add_argument("--skip_profiling", action="store_true", help="Skip profiling, run scaling benchmark only")
    args = parser.parse_args()

    if not os.path.exists(args.graph):
        print(f"Error: Graph file '{args.graph}' not found.")
        print(f"Please run from the directory containing the file, or provide --graph.")
        return

    g = load_graph_file(args.graph)

    if not args.skip_scaling:
        run_scaling_benchmark(g, args.sizes, args.generations, args.pop_size, args.worker_count)
        
    if not args.skip_profiling:
        run_profiling(g, args.generations, args.pop_size, args.profile_size, args.worker_count)


if __name__ == "__main__":
    main()
