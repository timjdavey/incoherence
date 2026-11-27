"""Deal the observations back out at random, for :meth:`Result.floor`."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ._entropy import Array, densities

#: A table of counts must hold fewer observations than this to be shuffled, a limit set by numpy.
MAX_DEALT = 10**9

#: Shuffles the observations between the trials and returns the new rows.
Deal = Callable[[np.random.Generator], Array]


def as_whole(table: Array) -> NDArray[np.int64] | None:
    """Return a table as whole-number counts, or ``None`` if it cannot be shuffled.

    Probabilities cannot, because they do not say how many observations a trial holds.
    """
    if (table != np.round(table)).any() or table.max() >= MAX_DEALT or table.sum() >= MAX_DEALT:
        return None
    return table.astype(np.int64)


def density_rows(groups: list[Array], k: float, points: int) -> Array:
    """Return one row of densities for each group of observations."""
    return np.stack([densities(group, k, points) for group in groups])


def deal_observations(
    trials: list[Array], to_rows: Callable[[list[Array]], Array], rng: np.random.Generator
) -> Array:
    """Deal the pooled observations out at random into trials of the same sizes."""
    cuts = np.cumsum([len(trial) for trial in trials])[:-1]
    return to_rows(np.split(rng.permutation(np.concatenate(trials)), cuts))


def deal_counts(table: NDArray[np.int64], rng: np.random.Generator) -> Array:
    """Deal the pooled counts out at random into trials of the same sizes.

    Each trial draws its observations from what is left of the pool, without
    putting any back. That gives what shuffling the observations one by one
    would give, without having to list them.
    """
    sizes = table.sum(axis=1)
    left = table.sum(axis=0)
    dealt = np.empty_like(table)
    for i, size in enumerate(sizes[:-1]):
        dealt[i] = rng.multivariate_hypergeometric(left, size)
        left = left - dealt[i]
    dealt[-1] = left
    return dealt / sizes[:, None]
