import random

def generate_adj_list_graph(output_file):
    import networkx as nx

    # If num_vertices and num_edges are not used, use the hardcoded values as per instruction
    G = nx.erdos_renyi_graph(20000, 0.005)

    with open(output_file, "w") as f:
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if neighbors:
                line = f"{node + 1} " + " ".join(str(n + 1) for n in neighbors) + "\n"
                f.write(line)

if __name__ == "__main__":
    generate_adj_list_graph("example-big.graph")
