from tau_community_detection import TauClustering, TauConfig

if __name__ == '__main__':
    graph_path = '/home/espl/TAU/tests/lfr_nodes10000_iter1.graph'
    config = TauConfig(verbose=True)
    tau = TauClustering(graph_path, population_size=64, max_generations=10, config=config)
    vertex_clustering, generation_stats = tau.run(track_stats=True)
