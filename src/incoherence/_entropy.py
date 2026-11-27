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
from scipy.signal import lfilter
from scipy.stats import entropy

Array = NDArray[np.float64]

#: Fewest reference points for the density estimate, and the number the papers used.
POINTS = 501

#: A continuous entropy below this is rounding error, so it counts as exactly 0.
_ROUNDING = 1e-12


@lru_cache(maxsize=2)
def grid(points: int = POINTS) -> Array:
    """Return the reference points for the density estimate.

    Observations are scaled to [0, 1] first, and the grid overhangs by half a
    unit on each side.

    Only the last two grids are kept, one for incoherence and one for
    cohesion. Cohesion's grid grows with the number of trials, so keeping more
    would hold a grid for every size of ensemble measured.
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


def densities(
    observations: Array, k: float, points: int = POINTS, weights: Array | None = None
) -> Array:
    """Density of the observations around each grid point (Incoherence paper, eq. 2).

    Parameters
    ----------
    observations
        One-dimensional values, already scaled to [0, 1].
    k
        How quickly an observation's influence falls away with distance.
    points
        The number of grid points. A higher ``k`` needs more, see :func:`points_for`.
    weights
        How much each observation counts. Leave as ``None`` to count them equally.

    Returns
    -------
    Array
        One density for each point of the grid.

    Notes
    -----
    The kernel shrinks by the same factor at every step along the grid. So each
    observation is added at the grid point on either side of it, and one pass
    in each direction carries its influence onwards. The cost grows with the
    observations plus the grid points, rather than with the two multiplied.
    """
    # Scaled so the largest is 1, or dividing by a tiny total can overflow.
    weights = None if weights is None else weights / weights.max()
    r = grid(points)
    decay = np.exp(-k * (r[-1] - r[0]) / (points - 1))
    above = np.searchsorted(r, observations, side="right")
    below = above - 1
    mass = 1.0 if weights is None else weights
    rising = np.asarray(
        np.bincount(above, mass * np.exp(-k * (r[above] - observations)), points),
        dtype=np.float64,
    )
    falling = np.asarray(
        np.bincount(below, mass * np.exp(-k * (observations - r[below])), points),
        dtype=np.float64,
    )
    total = _carry(rising, decay) + _carry(falling[::-1], decay)[::-1]
    return total / (len(observations) if weights is None else weights.sum())


def _carry(values: Array, decay: float) -> Array:
    """Add to each value the one before it, shrunk by ``decay`` at every step."""
    return np.asarray(lfilter([1.0], [1.0, -decay], values), dtype=np.float64)


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
    The result is 1 for an even spread. A U-shaped spread varies less than an
    even one, so it is clipped at 1. The result is 0 for a single repeated
    value at either end of the range. Elsewhere that value scores a little
    more, up to about 0.003 at ``k = 10`` and much more at a low ``k``.

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
    out = np.clip(1 - (variance - v0) / (v1 - v0), 0.0, 1.0)
    return np.where(out < _ROUNDING, 0.0, out)
