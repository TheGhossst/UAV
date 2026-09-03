"""UAV placement baselines (random, k-means). SCA lives in uavdt.sca."""

from uavdt.placement.kmeans import place_kmeans
from uavdt.placement.random import place_random

__all__ = ["place_kmeans", "place_random"]
