import time
import argparse
import sys
import igraph as ig

def main():
    parser = argparse.ArgumentParser(description='Benchmark TAU community detection')
    parser.add_argument('--version', choices=['old', 'new'], required=True, 
                        help='Which version of TAU to run: "old" (pip installed) or "new" (local path)')
    parser.add_argument('--graph', type=str, default='/home/bnet/hillelch/connectome/proofread_filtered.graph',
                        help='Path to the graph file')
    
    args = parser.parse_args()
    
    if args.version == 'new':
        # Add the local source directory to the beginning of sys.path
        import os
        sys.path.insert(0, os.path.abspath('/home/bnet/hillelch/TAU/src'))
        
    import tau_community_detection
    from tau_community_detection import TauClustering, TauConfig
    
    print(f"[{args.version.upper()} VERSION]")
    print(f"Loaded tau_community_detection from: {tau_community_detection.__file__}")
    
    config = TauConfig(verbose=True)
    
    print(f"Loading graph: {args.graph}")
    t0 = time.time()
    # Read graph using igraph just to be safe, though TauClustering can load it
    # We will pass the string path and let TauClustering load it
    
    clusterer = TauClustering(
        graph_source=args.graph,
        population_size=60,
        max_generations=10,
        config=config
    )
    
    print("Running clustering...")
    run_t0 = time.time()
    result = clusterer.run()
    run_time = time.time() - run_t0
    total_time = time.time() - t0
    
    print(f"Run completed in: {run_time:.3f} seconds")
    print(f"Total time (including init): {total_time:.3f} seconds")
    print(f"Resulting number of communities: {len(result)}")
    print("-" * 50)

if __name__ == "__main__":
    main()
