"""Runtime comparison between the packaged TAU implementation and the legacy script."""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np

from tau_community_detection import TauClustering, TauConfig


def _run_new(path: Path, population: int, generations: int, workers: int, seed: int | None) -> tuple[float, float | None]:
    config = TauConfig(
        population_size=population,
        max_generations=generations,
        worker_count=workers,
        random_seed=seed,
        reuse_worker_pool=False,
    )
    clustering = TauClustering(path, population_size=population, max_generation=generations, config=config)
    start = time.perf_counter()
    _, history = clustering.run()
    clustering.close()
    elapsed = time.perf_counter() - start
    final_modularity = history[-1] if history else None
    return elapsed, final_modularity


def _run_legacy(path: Path, population: int, generations: int, workers: int, seed: int | None) -> tuple[float, float | None]:
    from multiprocessing import Pool

    from tests import tau_v2_dev as legacy

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    legacy.pop = []
    legacy.G_ig = legacy.load_graph(str(path))
    legacy.GRAPH_PATH = str(path)
    legacy.POPULATION_SIZE = population
    legacy.MAX_GENERATIONS = generations
    legacy.N_WORKERS = workers
    legacy.PROBS = legacy.get_probabilities(np.arange(population))
    legacy.N_ELITE = max(1, int(legacy.p_elite * population))
    legacy.N_IMMIGRANTS = max(1, int(legacy.p_immigrants * population))
    legacy.SIM_SAMPLE_SIZE = 20000
    legacy.SIM_INDICES = (
        np.random.choice(legacy.G_ig.vcount(), legacy.SIM_SAMPLE_SIZE, replace=False)
        if legacy.G_ig.vcount() > legacy.SIM_SAMPLE_SIZE
        else None
    )

    pool = Pool(workers, initializer=legacy.init_worker, initargs=(legacy.GRAPH_PATH,))
    legacy.POOL = pool

    start = time.perf_counter()
    history: list[float] | None = None
    try:
        _, history = legacy.find_partition()
    finally:
        elapsed = time.perf_counter() - start
        pool.close()
        pool.join()

    final_modularity = history[-1] if history else None
    return elapsed, final_modularity


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TAU implementations on the same graph.")
    parser.add_argument("graph", type=Path, help="Path to adjacency list or weighted edgelist")
    parser.add_argument("--population", type=int, default=60, help="Population size")
    parser.add_argument("--generations", type=int, default=100, help="Maximum generations")
    parser.add_argument("--workers", type=int, default=-1, help="Number of worker processes")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cpu_cap = os.cpu_count() or 1
    if args.workers == -1:
        workers = min(cpu_cap, args.population)
    else:
        workers = max(1, min(cpu_cap, args.population, args.workers))
    print(f"Graph: {args.graph}")
    print(f"Population: {args.population}; Generations: {args.generations}; Workers: {workers}")

    legacy_time, legacy_mod = _run_legacy(args.graph, args.population, args.generations, workers, args.seed)
    print(f"Legacy script runtime: {legacy_time:.3f}s; final modularity: {legacy_mod}")

    modern_time, modern_mod = _run_new(args.graph, args.population, args.generations, workers, args.seed)
    print(f"Modern package runtime: {modern_time:.3f}s; final modularity: {modern_mod}")

    if legacy_time:
        speedup = legacy_time / modern_time if modern_time else float("inf")
        print(f"Speed ratio (legacy / modern): {speedup:.3f}x")


if __name__ == "__main__":
    main()
