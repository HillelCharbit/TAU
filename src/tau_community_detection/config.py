"""Configuration objects for the TAU clustering algorithm."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(slots=True)
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
