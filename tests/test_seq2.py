import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import igraph as ig
from tau_community_detection.algorithm import TauClustering, TauConfig
import tau_community_detection.algorithm as algo

algo.Pool = algo._SequentialPool

call_count = 0
original_leiden = ig.Graph.community_leiden

def mocked_leiden(self, *args, **kwargs):
    global call_count
    call_count += 1
    return original_leiden(self, *args, **kwargs)

ig.Graph.community_leiden = mocked_leiden

g = ig.Graph.Erdos_Renyi(100, 0.1)
config = TauConfig(population_size=5, max_generations=2)
tau = TauClustering(g, config.population_size, config.max_generations, config)
tau.run()
print(f"Total calls: {call_count}")
