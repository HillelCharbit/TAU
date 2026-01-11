"""Partition representation and worker helpers."""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
import sys
from typing import Optional, Sequence

import igraph as ig
import numpy as np

# Global worker state
_GRAPH: Optional[ig.Graph] = None
_n_iterations: int = 3
_resolution_parameter: float = 1.0
_WEIGHTS: Optional[list[float]] = None
_RNG: Optional[np.random.Generator] = None

MEMBERSHIP_DTYPE = np.int32


def configure_main(
    graph: ig.Graph,
    n_iterations: int,
    resolution_parameter: float,
    seed: Optional[int] = None,
) -> None:
    """Configure state for main process."""
    global _GRAPH, _n_iterations, _resolution_parameter, _WEIGHTS, _RNG
    _GRAPH = graph
    _n_iterations = n_iterations
    _resolution_parameter = resolution_parameter
    _WEIGHTS = list(graph.es["weight"]) if "weight" in graph.es.attributes() else None
    _RNG = np.random.default_rng(seed)

def init_worker(
    graph_path: str,
    n_iterations: int,
    resolution_parameter: float,
    is_weighted: bool,
    default_weight: float,
    seed: Optional[int],
) -> None:
    """Worker initializer — loads graph from path."""
    import os
    os.environ['MPLBACKEND'] = 'Agg'
    
    from .graph import load_graph_worker
    
    graph = load_graph_worker(graph_path, default_weight=default_weight, is_weighted=is_weighted)
    
    # Derive worker seed from rank
    worker_rank = 0
    proc = mp.current_process()
    if proc._identity:
        worker_rank = proc._identity[0] - 1
    
    configure_main(
        graph,
        n_iterations,
        resolution_parameter,
        None if seed is None else seed + worker_rank,
    )

def get_graph() -> ig.Graph:
    if _GRAPH is None:
        raise RuntimeError(
            "Graph not initialized in this process. Call configure_main or init_worker first."
        )
    return _GRAPH


def get_rng() -> np.random.Generator:
    global _RNG
    if _RNG is None:
        _RNG = np.random.default_rng()
    return _RNG


def _resolve_weights(graph: ig.Graph) -> Optional[list[float]]:
    if _WEIGHTS is None and "weight" not in graph.es.attributes():
        return None
    if graph is _GRAPH:
        return _WEIGHTS
    if "weight" in graph.es.attributes():
        return [float(w) for w in graph.es["weight"]]
    return None


_dataclass_kwargs = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**_dataclass_kwargs)
class Partition:
    """Represents a candidate clustering solution."""

    membership: np.ndarray
    n_comms: int
    fitness: Optional[float]
    _sample_fraction: float

    def __init__(self, sample_fraction: float = 0.5, init_membership: Optional[Sequence[int]] = None):
        graph = get_graph()
        rng = get_rng()
        self._sample_fraction = sample_fraction
        if init_membership is None:
            self.membership = self._initialize_membership(graph, rng, sample_fraction)
        else:
            self.membership = np.asarray(init_membership, dtype=MEMBERSHIP_DTYPE)
        self.n_comms = int(self.membership.max()) + 1 if len(self.membership) else 0
        self.fitness = None

    @classmethod
    def from_membership(
        cls,
        membership: np.ndarray,
        sample_fraction: float,
        n_comms: int,
        fitness: Optional[float] = None,
        copy_membership: bool = False,
    ) -> "Partition":
        instance = cls.__new__(cls)
        instance.membership = np.array(
            membership,
            dtype=MEMBERSHIP_DTYPE,
            copy=copy_membership,
        )
        instance._sample_fraction = sample_fraction
        instance.n_comms = int(n_comms)
        instance.fitness = fitness
        return instance

    def clone(self, copy_membership: bool = False, reset_fitness: bool = True) -> "Partition":
        fitness = None if reset_fitness else self.fitness
        return self.from_membership(
            self.membership,
            sample_fraction=self._sample_fraction,
            n_comms=self.n_comms,
            fitness=fitness,
            copy_membership=copy_membership,
        )

    @staticmethod
    def _initialize_membership(
        graph: ig.Graph, rng: np.random.Generator, sample_fraction: float
    ) -> np.ndarray:
        n_nodes = graph.vcount()
        n_edges = graph.ecount()
        sample_nodes = max(1, int(n_nodes * sample_fraction))
        sample_edges = max(1, int(n_edges * sample_fraction)) if n_edges else 0

        if rng.random() > 0.5 or sample_edges == 0:
            subset = rng.choice(n_nodes, size=sample_nodes, replace=False)
            subgraph = graph.subgraph(subset)
        else:
            # Sample edges with probability proportional to weight
            weights = _resolve_weights(graph)
            if weights is not None:
                weights_array = np.array(weights)
                p = weights_array / weights_array.sum()
                subset = rng.choice(n_edges, size=sample_edges, replace=False, p=p)
            else:
                subset = rng.choice(n_edges, size=sample_edges, replace=False)
            
            subgraph = graph.subgraph_edges(subset)

        membership = np.full(n_nodes, -1, dtype=MEMBERSHIP_DTYPE)
        sub_nodes = [vertex.index for vertex in subgraph.vs]
        sub_partition = subgraph.community_leiden(
            objective_function="modularity",
            resolution_parameter=_resolution_parameter,
            weights=_resolve_weights(subgraph),
        )
        local_membership = np.asarray(sub_partition.membership, dtype=MEMBERSHIP_DTYPE)
        membership[sub_nodes] = local_membership

        next_label = int(local_membership.max()) + 1 if len(local_membership) else 0
        unassigned = membership == -1
        membership[unassigned] = np.arange(next_label, next_label + np.count_nonzero(unassigned))
        return membership

    def optimize(self) -> "Partition":
        graph = get_graph()
        partition = graph.community_leiden(
            objective_function="modularity",
            initial_membership=self.membership,
            n_iterations=_n_iterations,
            resolution_parameter=_resolution_parameter,
            weights=_resolve_weights(graph),
        )
        self.membership = np.asarray(partition.membership, dtype=MEMBERSHIP_DTYPE)
        self.n_comms = int(self.membership.max()) + 1
        self.fitness = float(partition.modularity)
        return self

    def mutate(self) -> "Partition":
        graph = get_graph()
        rng = get_rng()
        membership = self.membership.copy()
        if rng.random() > 0.5:
            comm_id = int(rng.integers(0, self.n_comms))
            indices = np.where(membership == comm_id)[0]
            if len(indices) > 2:
                if len(indices) > 10 and rng.random() > 0.5:
                    self._newman_split(graph, membership, indices, comm_id)
                else:
                    self._random_split(rng, membership, indices)
        else:
            self._merge_connected_communities(graph, membership, rng)
        self.n_comms = max(1, int(membership.max()) + 1) if membership.size else 0
        self.membership = membership
        return self

    def _newman_split(
        self,
        graph: ig.Graph,
        membership: np.ndarray,
        indices: np.ndarray,
        comm_id: int,
    ) -> None:
        subgraph = graph.subgraph(indices.tolist())
        new_assignment = np.asarray(
            subgraph.community_leading_eigenvector(clusters=2).membership,
            dtype=MEMBERSHIP_DTYPE,
        )
        new_assignment[new_assignment == 0] = comm_id
        new_assignment[new_assignment == 1] = self.n_comms
        membership[membership == comm_id] = new_assignment

    def _random_split(
        self, rng: np.random.Generator, membership: np.ndarray, indices: np.ndarray
    ) -> None:
        if len(indices) == 0:
            return
        size = int(rng.integers(1, max(2, len(indices) // 2 + 1)))
        chosen = rng.choice(indices, size=size, replace=False)
        membership[chosen] = self.n_comms

    def _merge_connected_communities(
        self, graph: ig.Graph, membership: np.ndarray, rng: np.random.Generator
    ) -> None:
        edge_count = graph.ecount()
        if edge_count == 0:
            return
        size = min(10, edge_count)
        if size <= 0:
            return
        candidate_edges = rng.choice(edge_count, size=size, replace=False)
        for edge_idx in np.atleast_1d(candidate_edges):
            v1, v2 = graph.es[int(edge_idx)].tuple
            comm1, comm2 = membership[v1], membership[v2]
            if comm1 == comm2:
                continue
            membership[membership == comm1] = comm2
            membership[membership == self.n_comms - 1] = comm1
            self.n_comms = max(self.n_comms - 1, 1)
            break


def create_partition(sample_fraction: float) -> Partition:
    return Partition(sample_fraction=sample_fraction)


def optimize_partition(partition: Partition) -> Partition:
    return partition.optimize()


def mutate_partition(partition: Partition) -> Partition:
    return partition.mutate()
