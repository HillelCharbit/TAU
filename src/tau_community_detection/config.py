"""Configuration objects for the TAU clustering algorithm."""
from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Optional, Tuple


# ``dataclass(slots=True)`` requires Python 3.10+. Provide a graceful fallback
# so our tests can run under older interpreters in CI / sandbox.
_dataclass_kwargs = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**_dataclass_kwargs)
class TauConfig:
    """Hyper-parameters controlling the TAU evolutionary clustering algorithm."""

    population_size: int = 60
    max_generations: int = 500
    worker_count: Optional[int] = None
    elite_fraction: float = 0.1
    immigrant_fraction: float = 0.15
    selection_power: int = 5
    elite_similarity_threshold: float = 0.9
    stopping_generations: int = 10
    stopping_jaccard: float = 0.98
    sim_sample_size: Optional[int] = 20_000
    leiden_iterations: int = 3
    leiden_resolution: float = 1.0
    weight_attribute: Optional[str] = "weight"
    default_edge_weight: float = 1.0
    worker_chunk_size: Optional[int] = None
    reuse_worker_pool: bool = True
    sample_fraction_range: Tuple[float, float] = (0.2, 0.9)
    random_seed: Optional[int] = None

    def resolve_worker_count(self, population_size: int) -> int:
        from os import cpu_count

        candidate = self.worker_count or cpu_count() or 1
        return max(1, min(population_size, candidate))

    def resolve_elite_count(self) -> int:
        return max(1, int(self.elite_fraction * self.population_size))

    def resolve_immigrant_count(self) -> int:
        return max(1, int(self.immigrant_fraction * self.population_size))
