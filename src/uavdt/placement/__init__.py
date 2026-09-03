"""UAV placement baselines (no SCA / TD3 in this milestone)."""

from uavdt.placement.kmeans import place_kmeans
from uavdt.placement.random import place_random

__all__ = ["place_kmeans", "place_random"]
