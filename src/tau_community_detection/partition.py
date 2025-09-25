"""Partition representation and related multiprocessing helpers."""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Optional, Sequence

import igraph as ig
import numpy as np

from .graph import load_graph

_GRAPH: ig.Graph | None = None
_LEIDEN_ITERATIONS: int = 3
_LEIDEN_RESOLUTION: float = 1.0
_LEIDEN_WEIGHT_ATTRIBUTE: Optional[str] = None
_RNG: np.random.Generator | None = None

MEMBERSHIP_DTYPE = np.int32


def configure_shared_state(
    graph: ig.Graph,
    leiden_iterations: int,
    leiden_resolution: float,
    weight_attribute: Optional[str],
    seed: Optional[int] = None,
) -> None:
    """Configure global state for the current process (typically the main process)."""
    global _GRAPH, _LEIDEN_ITERATIONS, _LEIDEN_RESOLUTION, _LEIDEN_WEIGHT_ATTRIBUTE, _RNG
    _GRAPH = graph
    _LEIDEN_ITERATIONS = leiden_iterations
    _LEIDEN_RESOLUTION = leiden_resolution
    _LEIDEN_WEIGHT_ATTRIBUTE = (
        weight_attribute if weight_attribute and weight_attribute in graph.es.attributes() else None
    )
    _RNG = np.random.default_rng(seed)


def init_worker(
    graph_path: str,
    leiden_iterations: int,
    leiden_resolution: float,
    weight_attribute: Optional[str],
    default_weight: float,
    seed: Optional[int],
) -> None:
    """Worker initializer to lazily load the graph and RNG in each process."""
    graph = load_graph(
        Path(graph_path),
        weight_attribute=weight_attribute,
        default_weight=default_weight,
    )
    worker_rank = 0
    current = mp.current_process()
    # Pool worker identities start at 1; normalise back to 0 so the first
    # worker reuses the user-provided seed. This keeps runs reproducible while
    # still spacing out additional workers deterministically.
    if current._identity:  # type: ignore[attr-defined]
        worker_rank = current._identity[0] - 1
    else:
        suffix = current.name.rsplit("-", 1)
        if len(suffix) == 2 and suffix[1].isdigit():
            worker_rank = int(suffix[1]) - 1
    process_seed = None if seed is None else seed + worker_rank
    configure_shared_state(
        graph,
        leiden_iterations,
        leiden_resolution,
        weight_attribute,
        process_seed,
    )


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


def _resolve_weights(graph: ig.Graph) -> Optional[list[float]]:
    if _LEIDEN_WEIGHT_ATTRIBUTE and _LEIDEN_WEIGHT_ATTRIBUTE in graph.es.attributes():
        return [float(w) for w in graph.es[_LEIDEN_WEIGHT_ATTRIBUTE]]
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
            self.membership = self._initialise_membership(graph, rng, sample_fraction)
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

        membership = np.full(n_nodes, -1, dtype=MEMBERSHIP_DTYPE)
        sub_nodes = [vertex.index for vertex in subgraph.vs]
        sub_partition = subgraph.community_leiden(
            objective_function="modularity",
            resolution_parameter=_LEIDEN_RESOLUTION,
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
            n_iterations=_LEIDEN_ITERATIONS,
            resolution_parameter=_LEIDEN_RESOLUTION,
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
