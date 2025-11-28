
# import networkx as nx
# from tau_community_detection import TauClustering, TauConfig
# import graph_tool.all as gt

# def main():
#     # Example: Reading a .net file
#     # Replace 'your_file.net' with the path to your .net file
#     # graph_nx = nx.read_pajek('your_file.net')
#     # Or if you need to specify node types:
#     # graph_nx = nx.read_pajek('your_file.net', create_using=nx.Graph)
#     graph = gt.collection.ns["advogato"]
#     # print(f"Graph facts:")
#     # print(f"  Number of vertices: {graph.num_vertices()}")
#     # print(f"  Number of edges: {graph.num_edges()}")
#     # degrees = [v.out_degree() for v in graph.vertices()]
#     # print(f"  Mean degree: {sum(degrees)/len(degrees):.2f}")
#     # import random
#     # edges = list(graph.edges())
#     # e = random.choice(edges)

#     # src = int(e.source())
#     # tgt = int(e.target())

#     # w = graph.ep["weight"][e] if "weight" in graph.ep else None

#     # print(f"Sample edge: ({src}, {tgt})")
#     # print(f"  Weight: {w}")
    
#     import igraph as ig
#     # graph = nx.read_adjlist('/home/espl/TAU/tests/dblp.graph', nodetype=int)

#     graph = ig.Graph.from_networkx(graph)
#     # Check if graph is weighted by presence of a "weight" attribute
#     weights = graph.es["weight"] if "weight" in graph.es.attributes() else None
#     # Run Leiden algorithm with weights for correct modularity calculation
#     leiden_result = graph.community_leiden(objective_function='modularity', weights=weights)
#     leiden_modularity = leiden_result.modularity
#     leiden_communities = len(leiden_result)
#     print(f"Leiden - Modularity: {leiden_modularity:.6f}")
#     print(
#         f"Loaded as-22july06: nodes={graph.vcount()} edges={graph.ecount()}"
#     )

#     config = TauConfig(random_seed=42, stopping_generations=5)

#     tau = TauClustering(
#         graph,
#         population_size=10,
#         max_generations=30,
#         config=config,
#     )

#     membership, modularity_history = tau.run()
#     best_modularity = modularity_history[-1] if modularity_history else float("nan")
#     print(f"TauClustering - Best modularity: {best_modularity:.6f}")
#     print(f"TauClustering - Community count: {len(set(membership))}")


# if __name__ == "__main__":
#     main()
import graph_tool.all as gt
import igraph as ig
from tau_community_detection import TauClustering, TauConfig


def gt_to_igraph(g, undirected=True, weight_prop_name="weight"):
    """Convert a graph_tool Graph to an igraph Graph, preserving weights if present."""
    edges = []
    weights = []
    has_weight = weight_prop_name in g.ep
    
    # Collect edges and weights
    for e in g.edges():
        u = int(e.source())
        v = int(e.target())
        edges.append((u, v))
        if has_weight:
            weights.append(float(g.ep[weight_prop_name][e]))
    
    # Create igraph graph with correct directedness from the start
    # This avoids issues with to_undirected() returning None in some igraph versions
    if undirected:
        ig_g = ig.Graph(n=g.num_vertices(), edges=edges, directed=False)
    else:
        ig_g = ig.Graph(n=g.num_vertices(), edges=edges, directed=g.is_directed())
    
    # Add weights if present
    if has_weight:
        ig_g.es["weight"] = weights
    
    return ig_g


def main():
    # 1. Load advogato from Netzschleuder via graph-tool
    gt_graph = gt.collection.ns["advogato"]
    print("graph-tool advogato:")
    print(f"  vertices: {gt_graph.num_vertices()}")
    print(f"  edges   : {gt_graph.num_edges()}")
    print(f"  edge properties: {list(gt_graph.ep.keys())}")
    
    # 2. Convert to igraph (undirected + weighted)
    graph = gt_to_igraph(gt_graph, undirected=True, weight_prop_name="weight")
    
    # Debug check
    if graph is None:
        print("ERROR: gt_to_igraph returned None!")
        return
    
    print("\nigraph graph:")
    print(f"  vertices: {graph.vcount()}")
    print(f"  edges   : {graph.ecount()}")
    print(f"  directed: {graph.is_directed()}")
    
    # Check for weights
    has_weights = "weight" in graph.es.attributes()
    print(f"  weighted: {has_weights}")
    
    # 3. Leiden with weights
    weights = graph.es["weight"] if has_weights else None
    leiden_result = graph.community_leiden(
        objective_function="modularity",
        weights=weights,
    )
    print(f"\nLeiden - Modularity: {leiden_result.modularity:.6f}")
    print(f"Leiden - Community count: {len(leiden_result)}")
    
    # 4. TauClustering on the same igraph graph
    config = TauConfig(random_seed=42, stopping_generations=5)
    tau = TauClustering(
        graph,
        population_size=10,
        max_generations=30,
        config=config,
    )
    
    membership, modularity_history = tau.run()
    best_modularity = modularity_history[-1] if modularity_history else float("nan")
    print(f"\nTauClustering - Best modularity: {best_modularity:.6f}")
    print(f"TauClustering - Community count: {len(set(membership))}")


if __name__ == "__main__":
    main()