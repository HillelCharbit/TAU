import networkx as nx
import igraph as ig
from tau_community_detection import TauClustering, TauConfig

def main():
    cer_path = "/home/espl/TAU/tests/H_sapiens_weighted.graph"

    # --- FIX 1: NetworkX Reading ---
    # Explicitly tell NetworkX to read the 3rd column as a float weight
    G_nx = nx.read_edgelist(cer_path, nodetype=int, data=(('weight', float),))
    # Add verbose=True to see detailed logging in all TauClustering runs
    config = TauConfig(verbose = True)
    
    # --- FIX 2: igraph Reading ---
    # igraph.Read_Edgelist does not support weights directly. 
    # Use Read_Ncol instead, which is designed for "Source Target Weight" format.
    G_ig = ig.Graph.Read_Ncol(cer_path, names=True, weights="if_present")   

    print("--- Running Path ---")
    with TauClustering(cer_path, config=config, population_size=16, max_generations=2) as tau:
        clustering_tmp = tau.run()
    print(f"Q: {clustering_tmp.modularity}")

    print("--- Running NetworkX ---")
    with TauClustering(G_nx, config=config, population_size=16, max_generations=2) as tau:
        clustering_nx = tau.run()
    print(f"Q: {clustering_nx.modularity}")

    print("--- Running igraph ---")
    with TauClustering(G_ig, config=config, population_size=16, max_generations=2) as tau:
        clustering_ig = tau.run()
    print(f"Q: {clustering_ig.modularity}")

if __name__ == "__main__":
    main()
