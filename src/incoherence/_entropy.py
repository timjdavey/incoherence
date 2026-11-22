"""Entropy estimators.

Each estimator maps the rows of a ``(trials, points)`` array to one entropy per
row. A discrete row is a probability for each state. A continuous row is a
density at each point of a grid. Because both are plain rows, a pooled
distribution is a weighted mean of rows in either case.
"""

from functools import lru_cache
from math import ceil

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy

Array = NDArray[np.float64]

#: Fewest reference points for the density estimate, and the number the papers used.
POINTS = 501

#: Largest temporary array, in values, that the density estimate builds at once.
_BLOCK = 2**21


@lru_cache
def grid(points: int = POINTS) -> Array:
    """Return the reference points for the density estimate.

    Observations are scaled to [0, 1] first, and the grid overhangs by half a
    unit on each side.
    """
    out = np.linspace(-0.5, 1.5, points)
    out.setflags(write=False)
    return out


def points_for(k: float) -> int:
    """Return how many grid points the density estimate needs for this ``k``.

    An observation's influence falls away over a distance of ``1 / k``. The
    grid needs about ten points across that distance, or a value that falls
    between two points is partly missed. The default grid already gives that
    for ``k`` up to 25, so those results match the papers.
    """
    return max(POINTS, ceil(20 * k) + 1)


def shannon(pmfs: Array) -> Array:
    """Shannon entropy of each row, in bits (Incoherence paper, eq. 1)."""
    return np.asarray(entropy(pmfs, base=2, axis=-1), dtype=np.float64)


def densities(observations: Array, k: float, points: int = POINTS) -> Array:
    """Density of the observations around each grid point (Incoherence paper, eq. 2).

    Parameters
    ----------
    observations
        One-dimensional values, already scaled to [0, 1].
    k
        How quickly an observation's influence falls away with distance.
    points
        The number of grid points. A higher ``k`` needs more, see :func:`points_for`.

    Returns
    -------
    Array
        One density for each point of the grid.
    """
    r = grid(points)
    step = max(1, _BLOCK // points)
    total = np.zeros(points)
    for start in range(0, len(observations), step):
        chunk = observations[start : start + step]
        total += np.exp(-k * np.abs(r[:, None] - chunk[None, :])).sum(axis=1)
    return total / len(observations)


@lru_cache
def _variance_limits(k: float, power: float, points: int) -> tuple[float, float]:
    """Variance of the densities at maximum and minimum entropy (V0 and V1).

    V0 is for observations spread evenly over [0, 1], using the exact integral
    of the kernel. V1 is for every observation sitting on the same value.
    """
    r = grid(points)
    # Each branch only sees its own stretch of the grid, so a high k cannot overflow.
    within, under, over = np.clip(r, 0, 1), np.minimum(r, 0), np.maximum(r, 1)
    inside = (2 - np.exp(-k * within) - np.exp(-k * (1 - within))) / k
    below = np.exp(k * under) * (1 - np.exp(-k)) / k
    above = np.exp(-k * (over - 1)) * (1 - np.exp(-k)) / k
    uniform = np.where(r < 0, below, np.where(r > 1, above, inside))
    point = np.exp(-k * np.abs(r))
    return float((uniform**power).var()), float((point**power).var())


def density_entropy(rows: Array, k: float, power: float = 1.0) -> Array:
    """Continuous entropy of each row of densities (Incoherence paper, eq. 3).

    This is the estimator the papers call density variance. Densities that
    vary a lot across the grid mean clustered observations and low entropy.
    The result is 1 for an even spread and 0 for a single repeated value.

    Parameters
    ----------
    rows
        Densities from :func:`densities`, one row per distribution. The number
        of columns is the number of grid points.
    k
        The same ``k`` that produced the densities.
    power
        Applied to the densities before the variance is taken.
    """
    v0, v1 = _variance_limits(k, power, rows.shape[-1])
    variance = (rows**power).var(axis=-1)
    return np.clip(1 - (variance - v0) / (v1 - v0), 0.0, 1.0)
