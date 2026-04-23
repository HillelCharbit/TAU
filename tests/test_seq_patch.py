import igraph as ig
import tau_community_detection.algorithm as algo

algo.Pool = algo._SequentialPool # Force sequential execution

original_leiden = ig.Graph.community_leiden
call_count = 0

def mocked_leiden(self, *args, **kwargs):
    global call_count
    call_count += 1
    return original_leiden(self, *args, **kwargs)

ig.Graph.community_leiden = mocked_leiden

print(f"Patched! Pool is {algo.Pool}")


