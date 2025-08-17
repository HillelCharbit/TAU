import numpy as np
import networkx as nx
import igraph as ig
from sklearn.metrics.cluster import pair_confusion_matrix
import time
import os
import random
import argparse
from multiprocessing import Pool, set_start_method
import tempfile

# ---------------------- Multiprocessing Setup ----------------------
try:
    set_start_method('spawn')  # ensure Windows compatibility on Windows platforms
except RuntimeError:
    pass

# ---------------------- Global Variables ----------------------
G_ig = None           # Global igraph Graph
GRAPH_PATH = None     # Path to adjacency-list file for worker initializer
pop = []              # Current population of Partition instances
POOL = None           # Persistent multiprocessing Pool

# For optional approximate similarity via sub-sampling
SIM_SAMPLE_SIZE = None
SIM_INDICES = None

def init_worker(path, resolution=1.0):
    global G_ig, RESOLUTION
    G_ig = load_graph(path)
    RESOLUTION = resolution


class Partition:
    def __init__(self, sample_fraction=.5, init_partition=None):
        np.random.seed()
        self.n_nodes = G_ig.vcount()
        self.n_edges = G_ig.ecount()
        self.sample_size_nodes = int(self.n_nodes * sample_fraction)
        self.sample_size_edges = int(self.n_edges * sample_fraction)
        self.membership = []
        self.n_comms = 0
        self.fitness = None
        if init_partition is None:
            self.initialize_partition()
        else:
            self.membership = init_partition
            self.n_comms = len(np.unique(self.membership))

    def initialize_partition(self):
        if flip_coin():
            # sample nodes
            subsample = np.random.choice(self.n_nodes, size=self.sample_size_nodes, replace=False)
            subgraph = G_ig.subgraph(subsample)
        else:
            # sample edges
            subsample = np.random.choice(self.n_edges, size=self.sample_size_edges, replace=False)
            subgraph = G_ig.subgraph_edges(subsample)
        subsample_partition_memb = np.zeros(self.n_nodes) - 1
        subsample_nodes = [v.index for v in subgraph.vs]
        # leiden on subgraph
        subsample_subpartition = subgraph.community_leiden(objective_function='modularity', resolution_parameter=RESOLUTION)
        subsample_subpartition_memb = subsample_subpartition.membership
        subsample_partition_memb[subsample_nodes] = subsample_subpartition_memb
        first_available_comm_id = np.max(subsample_subpartition_memb) + 1
        arg_unassigned = subsample_partition_memb == -1
        subsample_partition_memb[arg_unassigned] = list(range(first_available_comm_id,
                                                              first_available_comm_id + sum(arg_unassigned)))
        self.membership = subsample_partition_memb.astype(int)
        self.n_comms = np.max(self.membership)+1

    def optimize(self):
        # leiden
        partition = G_ig.community_leiden(objective_function='modularity', initial_membership=self.membership,
                                          n_iterations=3, resolution_parameter=RESOLUTION)
        self.membership = partition.membership
        self.n_comms = np.max(self.membership) + 1
        self.fitness = partition.modularity
        return self

    def newman_split(self, indices, comm_id_to_split):
        # newman
        subgraph = G_ig.subgraph(indices)
        new_assignment = subgraph.community_leading_eigenvector(clusters=2).membership
        new_assignment[new_assignment == 0] = comm_id_to_split
        new_assignment[new_assignment == 1] = self.n_comms
        self.membership[self.membership == comm_id_to_split] = new_assignment

    def random_split(self, indices):
        size_to_split = min(1, np.random.choice(len(indices)//2))
        idx_to_split = np.random.choice(indices, size=size_to_split, replace=False)
        self.membership[idx_to_split] = self.n_comms

    def mutate(self):
        self.membership = np.array(self.membership)
        if flip_coin():
            # split a community
            comm_id_to_split = np.random.choice(self.n_comms)
            idx_to_split = np.where(self.membership == comm_id_to_split)[0]
            if len(idx_to_split) > 2:
                min_comm_size_newman = 10
                if len(idx_to_split) > min_comm_size_newman:
                    if flip_coin():
                        self.newman_split(idx_to_split, comm_id_to_split)
                    else:
                        self.random_split(idx_to_split)
                else:
                    self.random_split(idx_to_split)
                self.n_comms += 1
        else:
            # randomly merge two connected communities
            candidate_edges = random.choices(G_ig.es, k=10)
            for i, e in enumerate(candidate_edges):
                v1, v2 = e.tuple
                comm1, comm2 = self.membership[v1], self.membership[v2]
                if comm1 == comm2:
                    continue
                self.membership[self.membership == comm1] = comm2
                self.membership[self.membership == self.n_comms-1] = comm1
                self.n_comms -= 1
                break
        return self


def flip_coin():
    return random.uniform(0, 1) > .5


def load_graph(source):
    """
    Load a graph from various formats into an undirected igraph.Graph.

    Parameters:
        source (str | networkx.Graph | pandas.DataFrame): Input source path or object.

    Returns:
        igraph.Graph: The loaded undirected graph.
    """
    if isinstance(source, str):
        path = source.lower()

        # .graph format as adjacency list (e.g. "1 2 3" means node 1 connects to 2 and 3)
        if path.endswith(".graph"):
            nx_graph = nx.read_adjlist(path)
            mapping = dict(zip(nx_graph.nodes(), range(nx_graph.number_of_nodes())))
            nx_graph = nx.relabel_nodes(nx_graph, mapping)
            ig_graph = ig.Graph(len(nx_graph), list(zip(*list(zip(*nx.to_edgelist(nx_graph)))[:2])))
            return ig_graph

        # .csv format
        elif path.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(source)
            edges = list(zip(df.iloc[:, 0], df.iloc[:, 1]))
            return ig.Graph(edges=edges, directed=False)

        # .adjlist or .txt interpreted as networkx adjacency list
        elif path.endswith(".adjlist") or path.endswith(".txt"):
            nx_graph = nx.read_adjlist(source, create_using=nx.Graph(), nodetype=int)
            return load_graph(nx_graph)

        else:
            raise ValueError(f"Unsupported file format: {source}")

    # networkx graph object
    elif isinstance(source, nx.Graph):
        G = nx.convert_node_labels_to_integers(source)
        edges = list(G.edges())
        ig_graph = ig.Graph(edges=edges, directed=False)
        return ig_graph

    # pandas dataframe interpreted as edge list
    elif "pandas" in str(type(source)):
        edges = list(zip(source.iloc[:, 0], source.iloc[:, 1]))
        return ig.Graph(edges=edges, directed=False)

    else:
        raise TypeError("Unsupported input type for graph loading.")


def overlap(partition_memberships):
    consensus_partition = partition_memberships[0]
    n_nodes = len(consensus_partition)
    for i, partition in enumerate(partition_memberships[1:]):
        partition = partition
        cluster_id_mapping = {}
        c = 0
        for node_id in range(n_nodes):
            cur_pair = (consensus_partition[node_id], partition[node_id])
            if cur_pair not in cluster_id_mapping:
                cluster_id_mapping[cur_pair] = c
                c += 1
            consensus_partition[node_id] = cluster_id_mapping[cur_pair]
    return consensus_partition


def create_population(size_of_population):
    """Initialize a population of Partitions with random sample fractions."""
    fracs = np.random.uniform(0.2, 0.9, size_of_population)
    return POOL.map(Partition, fracs)


def get_probabilities(values):
    p = []
    values = np.max(values) + 1 - values
    denom = np.sum(values.astype(np.int64) ** SELECTION_POWER)
    for value in values:
        p.append(value ** SELECTION_POWER / denom)
    return p


def single_crossover(pair):
    partitions_overlap = overlap([pair[0].membership, pair[1].membership])
    single_offspring = Partition(init_partition=partitions_overlap)
    return single_offspring


def pop_crossover_and_immigration(n_offspring):
    pairs_to_cross = []
    as_is_offspring = []
    for i in range(n_offspring):
        idx1, idx2 = np.random.choice(len(pop), size=2, replace=False, p=PROBS)
        if flip_coin():
            pairs_to_cross.append([pop[idx1], pop[idx2]])
        else:
            as_is_offspring.append(pop[idx1])
    crossed_offspring = POOL.map(single_crossover, pairs_to_cross)
    offspring = crossed_offspring + as_is_offspring

    immigrants = create_population(size_of_population=N_IMMIGRANTS)

    return offspring, immigrants


def compute_partition_similarity(a, b):
    """
    Compute Jaccard similarity between two clusterings using pair_confusion_matrix.
    Returns a/(a + c + d) where a = same-cluster pairs in both.
    """
    b_, d, c, a_ = pair_confusion_matrix(a, b).flatten()
    return a_ / (a_ + c + d)


def _get_similarity(i, j):
    """
    Retrieve or compute (approximate) Jaccard similarity between pop[i] and pop[j].
    If SIM_INDICES is set, uses sub-sampled nodes for speed.
    Caches results in _sim_cache.
    """
    # Determine which membership slices to use
    if SIM_INDICES is not None:
        a = np.array(pop[i].membership)[SIM_INDICES]
        b = np.array(pop[j].membership)[SIM_INDICES]
    else:
        a = pop[i].membership
        b = pop[j].membership
    sim = compute_partition_similarity(a, b)
    return sim


def elitist_selection(threshold):
    """
    Incremental elitist selection: choose up to N_ELITE indices in pop
    whose Jaccard similarity to all previously chosen elites is ≤ threshold.
    Processes candidates in fitness order (pop sorted descending), then
    randomly fills if not enough qualify.
    """
    N = len(pop)
    if N_ELITE > N:
        raise ValueError("N_ELITE cannot exceed population size")

    elites = []
    remaining = set(range(N))
    # pop is assumed sorted by descending fitness
    for idx in range(N):
        if len(elites) >= N_ELITE:
            break
        candidate = idx
        # check novelty
        novel = True
        for e in elites:
            if _get_similarity(e, candidate) > threshold:
                novel = False
                break
        if novel:
            elites.append(candidate)
            remaining.discard(candidate)
    # random fill if needed
    if len(elites) < N_ELITE:
        fill = random.sample(list(remaining), N_ELITE - len(elites))
        elites.extend(fill)
    return elites


# Wrappers for POOL.map convenience
optimize_individual = Partition.optimize
mutate_individual = Partition.mutate


def find_partition():
    global pop
    last_best = None
    best_modularity_per_generation = []
    cnt_convergence = 0

    # Population initialization
    pop = create_population(size_of_population=POPULATION_SIZE)

    for generation_i in range(1, MAX_GENERATIONS+1):
        start_time = time.time()

        # 1. Optimize each individual
        pop = POOL.map(optimize_individual, pop)

        # 2. Record fitness & check convergence
        fits = [ind.fitness for ind in pop]
        best_idx = int(np.argmax(fits))
        best_fit = fits[best_idx]
        best_modularity_per_generation.append(best_fit)
        # convergence: consecutive gens above Jaccard threshold
        if last_best is not None:
            jacc = compute_partition_similarity(pop[best_idx].membership, last_best)
            if jacc >= stopping_criterion_jaccard:
                cnt_convergence += 1
            else:
                cnt_convergence = 0
        last_best = pop[best_idx].membership.copy()
        if cnt_convergence >= stopping_criterion_generations:
            break
        if generation_i == MAX_GENERATIONS:
            break

        # 3. Sort by fitness
        pop.sort(key=lambda x: x.fitness, reverse=True)
        # 4. Elitist selection
        elt_st = time.time()
        elite_idx = elitist_selection(elite_similarity_threshold)
        elt_rt = time.time() - elt_st
        elite = [pop[i] for i in elite_idx]

        # 5. Crossover + immigration
        crim_st = time.time()
        offspring, immigrants = pop_crossover_and_immigration(n_offspring=POPULATION_SIZE-N_ELITE-N_IMMIGRANTS)
        crim_rt = time.time() - crim_st

        # 6. Mutation
        offspring = POOL.map(mutate_individual, offspring)
        pop = elite + offspring + immigrants

        print(f'Generation {generation_i} Top fitness: {best_fit:.5f}; Average fitness: '
              f'{np.mean(fits):.5f}; Time per generation: {time.time() - start_time:.3f}; '
              f'convergence: {cnt_convergence} ; elt-runtime={elt_rt:.3f} ; crim-runtime={crim_rt:.3f}')

    # return best and modularity history
    return pop[0], best_modularity_per_generation


# globals and hyper-parameters
POPULATION_SIZE = 60
N_WORKERS = 60
MAX_GENERATIONS = 500
N_IMMIGRANTS = -1
N_ELITE = -1
SELECTION_POWER = 5
PROBS = []
RESOLUTION = 1.0
p_elite = .1
p_immigrants = .15
stopping_criterion_generations = 10
stopping_criterion_jaccard = .98
elite_similarity_threshold = .9

def run_clustering(graph, graph_name='unnamed_graph', size=60, max_generations=500, workers=-1, seed=None, resolution=1):
    global POPULATION_SIZE, MAX_GENERATIONS, GRAPH_PATH
    global N_WORKERS, PROBS, N_ELITE, N_IMMIGRANTS, SIM_INDICES, G_ig, POOL
    global RESOLUTION

    temp_path = None  # temp file path if needed

    # Convert NetworkX graph to .graph file if needed
    if isinstance(graph, nx.Graph):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".graph", mode='w')
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            line = ' '.join(map(str, [node] + neighbors))
            temp.write(line + '\n')
        temp.close()
        GRAPH_PATH = temp.name
        temp_path = temp.name
    elif isinstance(graph, str):
        GRAPH_PATH = graph
    else:
        raise TypeError("Input to run_clustering must be a file path or a NetworkX graph.")

    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Set parameters
    POPULATION_SIZE = max(10, size)
    MAX_GENERATIONS = max_generations
    cpus = os.cpu_count()
    N_WORKERS = min(cpus, POPULATION_SIZE) if workers == -1 else min(cpus, POPULATION_SIZE, workers)
    PROBS = get_probabilities(np.arange(POPULATION_SIZE))
    N_ELITE = int(p_elite * POPULATION_SIZE)
    N_IMMIGRANTS = int(p_immigrants * POPULATION_SIZE)
    RESOLUTION = resolution

    # Load graph in master process (for sampling) and in workers
    G_ig = load_graph(GRAPH_PATH)
    POOL = Pool(N_WORKERS, initializer=init_worker, initargs=(GRAPH_PATH, RESOLUTION))

    # Optional sampling indices
    SIM_SAMPLE_SIZE = 20000
    if G_ig.vcount() > SIM_SAMPLE_SIZE:
        SIM_INDICES = np.random.choice(G_ig.vcount(), SIM_SAMPLE_SIZE, replace=False)

    print(f'Main parameter values: pop_size={POPULATION_SIZE}, workers={N_WORKERS}, max_generations={MAX_GENERATIONS}')

    best_partition, mod_history = find_partition()
    np.save(f'TAU_partition_{graph_name}.npy', best_partition.membership)
    print("Best modularity:", best_partition.fitness)
    POOL.close()
    POOL.join()
    
    return best_partition.membership, best_partition.fitness

    # Clean up temporary file
    if temp_path is not None:
        os.remove(temp_path)
if __name__ == "__main__":
    G = nx.erdos_renyi_graph(100, 0.05)
    path = 'synthetic.graph'
    run_clustering(path, graph_name="erdos_test", size=30, max_generations=50)