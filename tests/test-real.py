
import networkx as nx
from tau_community_detection import TauClustering, TauConfig

def main():
    # Example: Reading a .net file
    # Replace 'your_file.net' with the path to your .net file
    # graph_nx = nx.read_pajek('your_file.net')
    # Or if you need to specify node types:
    # graph_nx = nx.read_pajek('your_file.net', create_using=nx.Graph)
    
    import igraph as ig
    graph = nx.read_adjlist('/home/espl/TAU/tests/dblp.graph', nodetype=int)

    graph = ig.Graph.from_networkx(graph)
    # Run Leiden algorithm
    leiden_result = graph.community_leiden(resolution=1.0)
    leiden_modularity = leiden_result.modularity
    leiden_communities = len(leiden_result)
    print(f"Leiden - Modularity: {leiden_modularity:.6f}")
    print(
        f"Loaded as-22july06: nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}"
    )

    config = TauConfig(random_seed=42, stopping_generations=5)

    tau = TauClustering(
        graph,
        population_size=10,
        max_generations=30,
        config=config,
    )

    membership, modularity_history = tau.run()
    best_modularity = modularity_history[-1] if modularity_history else float("nan")
    print(f"TauClustering - Best modularity: {best_modularity:.6f}")
    print(f"TauClustering - Community count: {len(set(membership))}")
    
    # Run Leiden algorithm
    leiden_result = graph.community_leiden(resolution=1.0)
    leiden_modularity = leiden_result.modularity
    leiden_communities = len(leiden_result)
    print(f"Leiden - Modularity: {leiden_modularity:.6f}")
    print(f"Leiden - Community count: {leiden_communities}")


if __name__ == "__main__":
    main()
