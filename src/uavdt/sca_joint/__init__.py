"""SCA-joint: Algorithm 1 plus a discrete a_ij/b_ij re-match.

Methodology probe. Frozen SCA (`uavdt.sca`, method=\"sca\") stays the
headline solver. Results from this package must be tagged
`method=\"sca_joint\"` and written to separately named files.
"""

from uavdt.sca_joint.algorithm import solve_sca_joint

__all__ = ["solve_sca_joint"]
