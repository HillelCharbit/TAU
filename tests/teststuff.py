#!/usr/bin/env python3
# simplest_sbm.py
import numpy as np
import networkx as nx
import igraph as ig

def simple_sbm_igraph(n_nodes: int, n_comms: int,
                      p_in: float, p_out: float,
                      weighted: bool = True,
                      seed: int = 42) -> ig.Graph:
    """
    Minimal SBM generator:
      - n_nodes: total nodes
      - n_comms: number of communities (equal sizes)
      - p_in: probability of intra-community edges
      - p_out: probability of inter-community edges
      - weighted: heavier weights intra, lighter weights inter
    Returns: igraph.Graph (undirected)
    """
    # equal community sizes
    base, r = divmod(n_nodes, n_comms)
    sizes = [base + (1 if i < r else 0) for i in range(n_comms)]

    # probability matrix
    P = np.full((n_comms, n_comms), p_out, dtype=float)
    np.fill_diagonal(P, p_in)

    # generate SBM (undirected, no self-loops)
    Gnx = nx.stochastic_block_model(sizes, P, seed=seed, directed=False, selfloops=False)

    # membership vector
    membership = np.concatenate([np.full(sz, cid, dtype=np.int32) for cid, sz in enumerate(sizes)])

    # edges as numpy
    edges = np.array(Gnx.edges(), dtype=np.int64)
    if edges.size == 0:
        g = ig.Graph(n=n_nodes, directed=False)
        g.vs["community"] = membership.tolist()
        g.es["weight"] = []
        return g

    # optional weights: intra > inter
    if weighted:
        same = membership[edges[:, 0]] == membership[edges[:, 1]]
        rng = np.random.default_rng(seed)
        w = np.empty(len(edges), dtype=np.float32)
        w[same]  = rng.uniform(0.7, 1.2, size=same.sum())  # heavier inside
        w[~same] = rng.uniform(0.1, 0.4, size=(~same).sum())  # lighter outside
        weights = w.tolist()
    else:
        weights = [1.0] * len(edges)

    # build igraph
    g = ig.Graph(n=n_nodes, edges=edges.tolist(), directed=False)
    g.vs["community"] = membership.tolist()
    g.es["weight"] = weights
    return g

if __name__ == "__main__":
    import igraph as ig
    import numpy as np

    g = simple_sbm_igraph(10000, 83, 0.1, 0.01)


    print(g.summary())
    import time

    start_time = time.time()
    partition = g.community_leiden(weights="weight", objective_function="modularity")
    elapsed_time = time.time() - start_time

    mod = partition.modularity
    print(f"Leiden modularity: {mod}")
    print(f"Time taken: {elapsed_time:.4f} seconds")
    
    from tau_community_detection import run_clustering
    
    # Build g_nx from g (convert igraph to networkx)
    G_nx = g.to_networkx()
    # Run tau_community_detection's run_clustering and time it
    start_time_tau = time.time()
    communities, mod_history = run_clustering(G_nx, graph_name='unnamed_graph', size=60, max_generations=200, workers=-1, seed=None, resolution=1)

    elapsed_time_tau = time.time() - start_time_tau


    print(f"tau_community_detection time taken: {elapsed_time_tau:.4f} seconds")
