import networkx as nx

from tau_community_detection import TauClustering


def run_tau_demo() -> None:
    G = nx.erdos_renyi_graph(20000, 0.005)
    clustering = TauClustering(G, population_size=60, max_generation=50)

    membership, mod_history = clustering.run()

    print(membership)
    print(mod_history)


if __name__ == "__main__":
    run_tau_demo()
