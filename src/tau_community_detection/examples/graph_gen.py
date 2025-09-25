import random

def generate_adj_list_graph(num_vertices, num_edges, output_file):
    # Use a set to avoid duplicate edges and self-loops
    edges = set()
    while len(edges) < num_edges:
        u = random.randint(0, num_vertices - 1)
        v = random.randint(0, num_vertices - 1)
        if u != v:
            edge = (min(u, v), max(u, v))
            if edge not in edges:
                edges.add(edge)

    # Build adjacency list
    adj = {i: [] for i in range(num_vertices)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Write to file in the format: node: neighbor1 neighbor2 ...
    with open(output_file, "w") as f:
        for node in range(num_vertices):
            neighbors = " ".join(str(neigh) for neigh in sorted(adj[node]))
            f.write(f"{node}: {neighbors}\n")

if __name__ == "__main__":
    generate_adj_list_graph(20000, 150000, "example-150k.graph")
