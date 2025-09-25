"""Run TAU clustering on the bundled 200k-node example graph."""

from __future__ import annotations

from pathlib import Path

from tau_community_detection import TauClustering, TauConfig


def main() -> None:
    graph_path = Path("src/tau_community_detection/examples/example-200k.graph")

    config = TauConfig(
        population_size=40,
        max_generations=20,
        worker_count=16,
        random_seed=7,
        sim_sample_size=20_000,
        reuse_worker_pool=False,
    )

    print(f"Running TAU on {graph_path}...")
    clustering = TauClustering(
        graph_path,
        population_size=config.population_size,
        max_generation=config.max_generations,
        config=config,
    )
    membership, history = clustering.run()
    clustering.close()

    final_modularity = history[-1] if history else None
    n_communities = len(set(membership))

    print(f"Final modularity: {final_modularity:.6f}" if final_modularity is not None else "No modularity history")
    print(f"Communities found: {n_communities}")


if __name__ == "__main__":
    main()
