"""Partition representation and related multiprocessing helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import igraph as ig
import numpy as np

from .graph import load_graph

_GRAPH: ig.Graph | None = None
_LEIDEN_ITERATIONS: int = 3
_RNG: np.random.Generator | None = None


def configure_shared_state(graph: ig.Graph, leiden_iterations: int, seed: Optional[int] = None) -> None:
    """Configure global state for the current process (typically the main process)."""
    global _GRAPH, _LEIDEN_ITERATIONS, _RNG
    _GRAPH = graph
    _LEIDEN_ITERATIONS = leiden_iterations
    _RNG = np.random.default_rng(seed)


def init_worker(graph_path: str, leiden_iterations: int, seed: Optional[int]) -> None:
    """Worker initializer to lazily load the graph and RNG in each process."""
    graph = load_graph(Path(graph_path))
    process_seed = None if seed is None else seed + (os.getpid() % 10_000)
    configure_shared_state(graph, leiden_iterations, process_seed)


def get_graph() -> ig.Graph:
    if _GRAPH is None:
        raise RuntimeError(
            "Graph not initialised in this process. Call configure_shared_state or init_worker first."
        )
    return _GRAPH


def get_rng() -> np.random.Generator:
    global _RNG
    if _RNG is None:
        _RNG = np.random.default_rng()
    return _RNG


@dataclass(slots=True)
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
            self.membership = self._initialise_membership(graph, rng, sample_fraction)
        else:
            self.membership = np.asarray(init_membership, dtype=int)
        self.n_comms = int(self.membership.max()) + 1 if len(self.membership) else 0
        self.fitness = None

    @staticmethod
    def _initialise_membership(
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
            subset = rng.choice(n_edges, size=sample_edges, replace=False)
            subgraph = graph.subgraph_edges(subset)

        membership = np.full(n_nodes, -1, dtype=int)
        sub_nodes = [vertex.index for vertex in subgraph.vs]
        sub_partition = subgraph.community_leiden(objective_function="modularity")
        local_membership = np.asarray(sub_partition.membership, dtype=int)
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
            n_iterations=_LEIDEN_ITERATIONS,
        )
        self.membership = np.asarray(partition.membership, dtype=int)
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
            dtype=int,
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
