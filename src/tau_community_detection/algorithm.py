"""High-level TAU clustering API."""
from __future__ import annotations

import multiprocessing
from typing import Iterable, Literal, Optional, Sequence, Union, overload
import logging
import time
import sys
import warnings

import igraph as ig
import networkx as nx
import numpy as np
from sklearn.metrics.cluster import pair_confusion_matrix

from .config import TauConfig
from .graph import load_graph
from .partition import (
    Partition,
    configure_main,
    create_partition,
    get_graph,
    init_worker,
    mutate_partition,
    optimize_partition,
)

_CROSSOVER_PROBABILITY = 0.5
_logger = logging.getLogger(__name__)


# LokyPool wrapper: use loky executor when available to avoid importing
# user's __main__ module in spawned worker processes (helps on Windows).
class LokyPool:
    def __init__(self, processes, initializer=None, initargs=()):
        try:
            from loky.process_executor import LokyProcessPoolExecutor
        except Exception as exc:
            raise ImportError("loky unavailable") from exc
        # Loky supports initializer/initargs similar to multiprocessing
        self._executor = LokyProcessPoolExecutor(max_workers=processes, initializer=initializer, initargs=initargs)

    def map(self, func, iterable, chunksize: int = 1):
        return list(self._executor.map(func, iterable))

    def map_async(self, func, iterable, chunksize: int = 1):
        futures = [self._executor.submit(func, item) for item in iterable]

        class _AsyncResult:
            def __init__(self, futures):
                self._futures = futures

            def get(self, timeout=None):
                return [f.result(timeout) for f in self._futures]

        return _AsyncResult(futures)

    def close(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    def join(self) -> None:
        try:
            self._executor.shutdown(wait=True)
        except Exception:
            pass


def _is_interactive_environment() -> bool:
    """Detect if running in Jupyter, IPython, or similar interactive environment."""
    try:
        get_ipython()  # noqa: F821
        return True
    except NameError:
        return False


class _SequentialPool:
    """Lightweight stand-in when multiprocessing pools cannot be created."""

    def __init__(self, *args, **kwargs):
        pass

    def map(self, func, iterable, chunksize: int = 1):  # noqa: D401 - simple sequential map
        return [func(item) for item in iterable]

    def map_async(self, func, iterable, chunksize: int = 1):
        items = list(iterable)

        class _AsyncResult:
            def get(self, timeout=None):
                return [func(item) for item in items]

        return _AsyncResult()

    def close(self) -> None:
        pass

    def join(self) -> None:
        pass


Pool = Union[LokyPool, _SequentialPool, "multiprocessing.pool.Pool"]


def _overlap_memberships(memberships: Iterable[np.ndarray]) -> tuple[np.ndarray, int]:
    """Build a consensus labelling for the provided membership arrays."""

    iterator = iter(memberships)
    consensus = np.array(next(iterator), copy=True)
    dtype = consensus.dtype
    for membership in iterator:
        member_arr = np.asarray(membership, dtype=dtype)
        if consensus.size == 0:
            consensus = member_arr.copy()
            continue
        base_codes = np.unique(consensus, return_inverse=True)[1]
        other_codes = np.unique(member_arr, return_inverse=True)[1]
        pair_codes = (base_codes.astype(np.int64) << 32) | other_codes.astype(np.int64)
        consensus = np.unique(pair_codes, return_inverse=True)[1].astype(dtype, copy=False)
    label_count = int(consensus.max()) + 1 if consensus.size else 0
    return consensus, label_count


def _crossover_pair(args: tuple[np.ndarray, np.ndarray, float]) -> Partition:
    membership_a, membership_b, sample_fraction = args
    membership, n_comms = _overlap_memberships((membership_a, membership_b))
    return Partition.from_membership(
        membership,
        sample_fraction=sample_fraction,
        n_comms=n_comms,
        fitness=None,
        copy_membership=False,
    )

class TauClustering:
    """Evolutionary community detection for large graphs."""
    
    def __init__(
        self,
        graph_source: ig.Graph | nx.Graph | str,
        config: Optional[TauConfig] = None,
    ):
        self._pool = None  # LokyPool | multiprocessing.Pool | _SequentialPool
        self._pool_processes: Optional[int] = None
        self.config = config or TauConfig()

        # Single load_graph call — returns (graph, is_weighted, path_for_workers)
        self.graph, self._resolved_is_weighted, self._graph_path = load_graph(
            graph_source,
            weight_attr="weight",
            default_weight=1.0,
            is_weighted=self.config.is_weighted,
        )
        self.config.is_weighted = self._resolved_is_weighted  # update config with resolved value

        configure_main(
            self.graph,
            self.config.n_iterations,
            self.config.resolution_parameter,
            self.config.random_seed,
        )
        
        self.rng = np.random.default_rng(self.config.random_seed)
        self.sim_indices = self._init_similarity_indices()
        self.selection_probs = self._selection_probabilities(self.config.population_size)
        
        self._log(
                f"Main parameter values: pop_size={self.config.population_size}, "
                f"workers={self.config.resolve_worker_count()}, "
                f"max_generations={self.config.max_generations}"
        )

    @overload
    def run(self, track_stats: Literal[False] = ...) -> ig.VertexClustering: ...
    @overload
    def run(self, track_stats: Literal[True]) -> tuple[ig.VertexClustering, list[dict[str, float]]]: ...
    def run(self, track_stats: bool = False):
        worker_count = self.config.resolve_worker_count()
        chunk_size = self._resolve_chunk_size(worker_count)
        pool = self._ensure_pool(worker_count)
        elite_count = min(self.config.resolve_elite_count(), self.config.population_size)
        immigrant_count = min(
            self.config.resolve_immigrant_count(),
            max(0, self.config.population_size - elite_count - 1),
        )
        offspring_count = max(0, self.config.population_size - elite_count - immigrant_count)

        generation_stats: list[dict[str, float]] | None = [] if track_stats else None
        last_best_membership: Optional[np.ndarray] = None
        convergence_streak = 0
        best_partition: Optional[Partition] = None

        self._log(f"Creating population with {self.config.population_size} individuals...")
        population = self._create_population(pool, self.config.population_size, chunk_size)
        self._log("Population created")

        
        original_names = (
            list(self.graph.vs["name"]) if "name" in self.graph.vs.attributes() else None
        )
        for generation in range(1, self.config.max_generations + 1):
            start_time = time.perf_counter()
            
            # Queue optimize tasks
            optimize_async = pool.map_async(optimize_partition, population, chunksize=chunk_size)

            # Queue immigrant tasks simultaneously allowing idle workers to pick them up
            immigrant_async = None
            if immigrant_count > 0:
                low, high = self.config.sample_fraction_range
                fractions = self.rng.uniform(low, high, immigrant_count)
                immigrant_async = pool.map_async(create_partition, fractions.tolist(), chunksize=chunk_size)

            # Block until optimization is done
            optimized = optimize_async.get()
            population[:] = optimized

            fitnesses = np.array([p.fitness if p.fitness is not None else float("-inf") for p in population])
            avg_fitness = float(np.mean(fitnesses)) if fitnesses.size else float("-inf")
            best_idx = int(np.argmax(fitnesses))
            best_partition = population[best_idx]
            best_modularity = float(best_partition.fitness or -np.inf)
            if last_best_membership is not None:
                jaccard = self._similarity_arrays(best_partition.membership, last_best_membership)
                if jaccard >= self.config.stopping_jaccard:
                    convergence_streak += 1
                else:
                    convergence_streak = 0
            last_best_membership = best_partition.membership.copy()

            should_stop = (
                convergence_streak >= self.config.stopping_generations
                or generation >= self.config.max_generations
            )

            if should_stop:
                # Record the final generation's stats before exiting
                gen_elapsed = time.perf_counter() - start_time
                self._log(
                    f"Generation {generation} Top fitness: {best_modularity:.5f}; Average fitness: "
                    f"{avg_fitness:.5f}; Time per generation: {gen_elapsed:.3f}; "
                    f"convergence: {convergence_streak}"
                )
                if track_stats and generation_stats is not None:
                    generation_stats.append(
                        {
                            "generation": generation,
                            "top_fitness": best_modularity,
                            "average_fitness": avg_fitness,
                            "time_per_generation": gen_elapsed,
                            "convergence": convergence_streak,
                            "elite_runtime": 0.0,
                            "crossover_runtime": 0.0,
                        }
                    )
                break

            population.sort(key=lambda part: part.fitness or float("-inf"), reverse=True)
            elite_start = time.perf_counter()
            elite_indices = self._elitist_selection(population, self.config.elite_similarity_threshold, elite_count)
            elite_elapsed = time.perf_counter() - elite_start
            elites = [population[i] for i in elite_indices]

            crossover_start = time.perf_counter()
            offspring = self._produce_offspring(pool, chunk_size, population, offspring_count)
            immigrants = immigrant_async.get() if immigrant_async is not None else []
            crossover_elapsed = time.perf_counter() - crossover_start

            if offspring:
                mutated_offspring = pool.map(mutate_partition, offspring, chunksize=chunk_size)
                offspring = mutated_offspring

            population[:] = elites
            population.extend(offspring)
            population.extend(immigrants)

            gen_elapsed = time.perf_counter() - start_time
            self._log(
                f"Generation {generation} Top fitness: {best_modularity:.5f}; Average fitness: "
                f"{avg_fitness:.5f}; Time per generation: {gen_elapsed:.3f}; "
                f"convergence: {convergence_streak}; elite_runtime={elite_elapsed:.3f}; crossover_runtime={crossover_elapsed:.3f}"
            )
            if track_stats and generation_stats is not None:
                generation_stats.append(
                    {
                        "generation": generation,
                        "top_fitness": best_modularity,
                        "average_fitness": avg_fitness,
                        "time_per_generation": gen_elapsed,
                        "convergence": convergence_streak,
                        "elite_runtime": elite_elapsed,
                        "crossover_runtime": crossover_elapsed,
                    }
                )

        if best_partition is None:
            raise RuntimeError("TAU clustering failed to produce any solution.")

        membership_result = best_partition.membership.astype(np.int64, copy=True)
        original_membership = None
        if original_names is not None and len(original_names) == len(membership_result):
            original_membership = {
                original_names[i]: int(membership_result[i]) for i in range(len(membership_result))
            }

        clustering_result = ig.VertexClustering(
            self.graph,
            membership=membership_result.tolist(),
        )
        if "weight" in self.graph.es.attributes():
            clustering_result._modularity = float(
                self.graph.modularity(
                    membership_result.tolist(),
                    weights=self.graph.es["weight"],
                )
            )
            clustering_result._modularity_dirty = False
        if original_membership is not None:
            clustering_result.original_membership = original_membership
        if track_stats and generation_stats is not None:
            return clustering_result, generation_stats
        return clustering_result


    def __enter__(self) -> "TauClustering":
        return self

    def __exit__(self, *_) -> None:
        self._shutdown_pool()
        
    def __del__(self):
        self._shutdown_pool()

    def _log(self, message: str) -> None:
        if self.config.verbose:
            _logger.info(message)

    
    def _create_population(self, pool: Pool, size: int, chunk_size: int) -> list[Partition]:
        if size <= 0:
            return []
        low, high = self.config.sample_fraction_range
        fractions = self.rng.uniform(low, high, size)
        return pool.map(create_partition, fractions.tolist(), chunksize=chunk_size)

    def _produce_offspring(
        self,
        pool: Pool,
        chunk_size: int,
        population: Sequence[Partition],
        count: int,
    ) -> list[Partition]:
        if count <= 0:
            return []
        offspring: list[Partition] = []
        crossover_jobs: list[tuple[np.ndarray, np.ndarray, float]] = []
        pop_size = len(population)
        for _ in range(count):
            parents = self.rng.choice(pop_size, size=2, replace=False, p=self.selection_probs)
            parent_a, parent_b = population[int(parents[0])], population[int(parents[1])]
            if self.rng.random() > _CROSSOVER_PROBABILITY:
                sample_fraction = (parent_a._sample_fraction + parent_b._sample_fraction) / 2.0
                crossover_jobs.append(
                    (parent_a.membership, parent_b.membership, sample_fraction)
                )
            else:
                offspring.append(parent_a.clone(copy_membership=False, reset_fitness=True))
        if crossover_jobs:
            offspring.extend(pool.map(_crossover_pair, crossover_jobs, chunksize=chunk_size))
        return offspring

    def _elitist_selection(
        self,
        population: Sequence[Partition],
        threshold: float,
        elite_count: int,
    ) -> list[int]:
        if elite_count >= len(population):
            return list(range(len(population)))

        elites: list[int] = []
        for idx, candidate in enumerate(population):
            if len(elites) >= elite_count:
                break
            if all(self._similarity(population[e], candidate) <= threshold for e in elites):
                elites.append(idx)

        if len(elites) < elite_count:
            remaining = [idx for idx in range(len(population)) if idx not in elites]
            if remaining:
                fill = self.rng.choice(
                    remaining,
                    size=min(len(remaining), elite_count - len(elites)),
                    replace=False,
                )
                elites.extend(int(i) for i in np.atleast_1d(fill))
        return elites


    def _ensure_pool(self, worker_count: int) -> Pool:
        if self._pool is not None and self._pool_processes == worker_count:
            return self._pool
        self._shutdown_pool()
        
        initargs = (
            self._graph_path,  # always a path now
            self.config.n_iterations,
            self.config.resolution_parameter,
            self._resolved_is_weighted,
            1.0,
            self.config.random_seed,
        )
        
        # Attempt parallel execution
        if worker_count > 1:
            try:
                # Prefer loky-based pool to avoid re-importing user's __main__ on spawn
                try:
                    self._pool = LokyPool(worker_count, initializer=init_worker, initargs=initargs)
                    self._pool_processes = worker_count
                    return self._pool
                except ImportError:
                    # loky not available; fall back to multiprocessing spawn pool
                    self._pool = multiprocessing.get_context("spawn").Pool(
                        worker_count, initializer=init_worker, initargs=initargs
                    )
                    self._pool_processes = worker_count
                    return self._pool
            except (PermissionError, RuntimeError, ValueError, OSError) as exc:
                # Common failures in Windows/Jupyter/macOS interactive environments
                if _is_interactive_environment():
                    warnings.warn(
                        "Parallel execution unavailable in interactive environment. "
                        "For Jupyter on Windows, consider:\n"
                        "  1. Using 'if __name__ == \"__main__\":' in scripts\n"
                        "  2. Installing loky: pip install loky\n"
                        "  3. Running with worker_count=1\n"
                        "Falling back to sequential processing.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                else:
                    self._log(f"Multiprocessing pool creation failed: {exc}. Falling back to sequential execution.")

                # Fall back to sequential execution
                self._pool = _SequentialPool()
                self._pool_processes = 1
                return self._pool

        # Single worker requested
        self._log("Using sequential execution (1 worker)")
        self._pool = _SequentialPool()
        self._pool_processes = 1
        return self._pool

    def _shutdown_pool(self) -> None:
        if getattr(self, "_pool", None) is None:
            return
        self._pool.close()
        self._pool.join()
        self._pool = None
        self._pool_processes = None

    def _resolve_chunk_size(self, worker_count: int) -> int:
        if worker_count <= 0:
            return 1
        return max(1, (self.config.population_size + worker_count - 1) // worker_count)


    def _similarity(self, a: Partition, b: Partition) -> float:
        return self._similarity_arrays(a.membership, b.membership)

    def _similarity_arrays(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.sim_indices is not None:
            a = a[self.sim_indices]
            b = b[self.sim_indices]
        tn, fn, fp, tp = pair_confusion_matrix(a, b).flatten()
        denominator = tp + fp + fn
        return float(tp / denominator) if denominator else 1.0

    def _selection_probabilities(self, population_size: int) -> np.ndarray:
        if population_size <= 0:
            return np.array([], dtype=float)
        indices = np.arange(population_size)
        max_val = int(indices[-1]) if population_size else 0
        scaled = max_val + 1 - indices
        weights = np.power(scaled.astype(float), self.config.selection_power)
        weights_sum = weights.sum()
        if weights_sum == 0:
            return np.full(population_size, 1.0 / population_size)
        return weights / weights_sum

    def _init_similarity_indices(self) -> Optional[np.ndarray]:
        sample_size = self.config.sim_sample_size
        graph = get_graph()
        if sample_size is None or graph.vcount() <= sample_size:
            return None
        return self.rng.choice(graph.vcount(), size=sample_size, replace=False)
