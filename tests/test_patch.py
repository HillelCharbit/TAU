import igraph as ig

count = 0
original_leiden = ig.Graph.community_leiden

def mocked_leiden(self, *args, **kwargs):
    global count
    count += 1
    return original_leiden(self, *args, **kwargs)

ig.Graph.community_leiden = mocked_leiden

g = ig.Graph.Erdos_Renyi(10, 0.5)
g.community_leiden(objective_function="modularity")

print("Leiden called:", count)
