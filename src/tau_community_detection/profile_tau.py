"""Performance profiling helper for the TAU community detection pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tau_community_detection import TauClustering, TauConfig  # noqa: E402


def _build_graph(node_count: int, edge_probability: float) -> nx.Graph:
    return nx.erdos_renyi_graph(node_count, edge_probability, seed=42)


def _collect_profile(graph: nx.Graph, config: TauConfig, output: Path) -> None:
    import cProfile
    import pstats
    import tracemalloc
    from tau_community_detection.algorithm import TauClustering as _TauClustering

    profiler = cProfile.Profile()
    tracemalloc.start()

    clustering = _TauClustering(graph, population_size=config.population_size, max_generation=config.max_generations, config=config)

    profiler.enable()
    tracemalloc.reset_peak()
    clustering.run()
    profiler.disable()
    clustering.close()

    current, peak = tracemalloc.get_traced_memory()
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats(pstats.SortKey.CUMULATIVE)

    sorted_stats = sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:30]
    func_labels: list[str] = []
    func_times: list[float] = []
    for (filename, line, func_name), stat in sorted_stats:
        cum_time = stat[3] * 1000.0
        func_labels.append(f"{Path(filename).name}:{line}\n{func_name}")
        func_times.append(cum_time)

    mem_stats = snapshot.statistics("lineno")[:15]
    mem_labels = [f"{Path(stat.traceback[0].filename).name}:{stat.traceback[0].lineno}" for stat in mem_stats]
    mem_sizes = [stat.size / (1024 * 1024) for stat in mem_stats]

    _render_figure(func_labels, func_times, mem_labels, mem_sizes, current, peak, output)


def _render_figure(
    func_labels: Iterable[str],
    func_times: Iterable[float],
    mem_labels: Iterable[str],
    mem_sizes: Iterable[float],
    current_memory: int,
    peak_memory: int,
    output: Path,
) -> None:
    func_labels = list(func_labels)[:12]
    func_times = list(func_times)[:12]
    mem_labels = list(mem_labels)[:12]
    mem_sizes = list(mem_sizes)[:12]

    plt.style.use("seaborn-v0_8")
    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle("TAU Profiling Snapshot", fontsize=16, fontweight="bold")

    y_pos = range(len(func_labels))
    ax_time.barh(list(y_pos), func_times, color="#1f77b4")
    ax_time.set_yticks(list(y_pos), func_labels)
    ax_time.invert_yaxis()
    ax_time.set_xlabel("Cumulative time (ms)")
    ax_time.set_title("Top functions by cumulative time")

    x_pos = range(len(mem_labels))
    ax_mem.bar(list(x_pos), mem_sizes, color="#2ca02c")
    ax_mem.set_xticks(list(x_pos), mem_labels, rotation=45, ha="right")
    ax_mem.set_ylabel("Allocated memory (MB)")
    ax_mem.set_title("Top lines by allocated memory")

    footer = (
        f"Current memory: {current_memory / (1024*1024):.2f} MB | "
        f"Peak memory: {peak_memory / (1024*1024):.2f} MB"
    )
    fig.text(0.5, 0.02, footer, ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Profiling figure written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile TAU clustering runtime and memory usage.")
    parser.add_argument("--nodes", type=int, default=7000, help="Number of nodes for ER graph")
    parser.add_argument("--prob", type=float, default=0.005, help="Edge probability for ER graph")
    parser.add_argument("--population", type=int, default=24, help="TAU population size")
    parser.add_argument("--generations", type=int, default=40, help="Maximum generations")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profiling") / "tau_profile.png",
        help="Destination for the generated figure",
    )
    args = parser.parse_args()

    graph = _build_graph(args.nodes, args.prob)
    config = TauConfig(
        population_size=args.population,
        max_generations=args.generations,
        weight_attribute=None,
        sim_sample_size=min(10_000, args.nodes),
    )

    _collect_profile(graph, config, args.output)


if __name__ == "__main__":
    main()
