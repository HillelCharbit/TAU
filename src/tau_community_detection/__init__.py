"""Public API for the TAU clustering package.

High-level, user-friendly community detection for large graphs.
"""
from .algorithm import TauClustering
from .api import run_clustering
from .config import TauConfig

__all__ = ["run_clustering", "TauClustering", "TauConfig"]

__version__ = "1.3.3"

