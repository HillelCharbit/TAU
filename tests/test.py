from pathlib import Path
import sys

import networkx as nx

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from tau_community_detection import TauClustering


def run_tau_demo() -> None:
    G = nx.erdos_renyi_graph(20000, 0.005)
    clustering = TauClustering(G, population_size=60, max_generation=50)

    membership, mod_history = clustering.run()

    print(membership)
    print(mod_history)


if __name__ == "__main__":
    run_tau_demo()
