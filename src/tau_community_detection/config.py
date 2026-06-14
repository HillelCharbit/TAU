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
    max_generations: int = 20
    resolution_parameter: float = 1.0
    random_seed: Optional[int] = None
    verbose: bool = False
    n_iterations: int = 3
    worker_count: Optional[int] = None
    elite_fraction: float = 0.1
    immigrant_fraction: float = 0.15
    selection_power: int = 5
    elite_similarity_threshold: float = 0.9
    stopping_generations: int = 10
    stopping_jaccard: float = 0.98
    sim_sample_size: Optional[int] = 20_000
    is_weighted: Optional[bool] = None
    sample_fraction_range: Tuple[float, float] = (0.2, 0.9)

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            raise ValueError(f"population_size must be > 0, got {self.population_size}")
        if self.max_generations <= 0:
            raise ValueError(f"max_generations must be > 0, got {self.max_generations}")
        if not (0 < self.elite_fraction <= 1):
            raise ValueError(f"elite_fraction must be in (0, 1], got {self.elite_fraction}")
        if not (0 < self.immigrant_fraction <= 1):
            raise ValueError(f"immigrant_fraction must be in (0, 1], got {self.immigrant_fraction}")
        if self.selection_power <= 0:
            raise ValueError(f"selection_power must be > 0, got {self.selection_power}")
        if not (0 <= self.elite_similarity_threshold <= 1):
            raise ValueError(f"elite_similarity_threshold must be in [0, 1], got {self.elite_similarity_threshold}")
        if not (0 <= self.stopping_jaccard <= 1):
            raise ValueError(f"stopping_jaccard must be in [0, 1], got {self.stopping_jaccard}")
        if self.n_iterations <= 0:
            raise ValueError(f"n_iterations must be > 0, got {self.n_iterations}")
        if self.stopping_generations <= 0:
            raise ValueError(f"stopping_generations must be > 0, got {self.stopping_generations}")
        if self.resolution_parameter <= 0:
            raise ValueError(f"resolution_parameter must be > 0, got {self.resolution_parameter}")
        low, high = self.sample_fraction_range
        if not (0 < low <= high <= 1):
            raise ValueError(
                f"sample_fraction_range must satisfy 0 < low <= high <= 1, got {self.sample_fraction_range}"
            )

    def resolve_worker_count(self) -> int:
        from os import cpu_count
        max_parallel_tasks = self.population_size + self.resolve_immigrant_count()
        candidate = self.worker_count or cpu_count() or 1
        return max(1, min(max_parallel_tasks, candidate))

    def resolve_elite_count(self) -> int:
        return max(1, int(self.elite_fraction * self.population_size))

    def resolve_immigrant_count(self) -> int:
        return max(1, int(self.immigrant_fraction * self.population_size))
