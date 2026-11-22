"""The equations from the papers, as pure functions.

Every function works on rows and an entropy estimator, so the same code serves
discrete and continuous data. See ``_entropy`` for what a row is.
"""

from collections.abc import Callable

import numpy as np

from ._entropy import _BLOCK, Array, densities, density_entropy, points_for

Entropy = Callable[[Array], Array]

#: Settings for the entropy of the pairwise divergences (Cohesion paper, eq. 3 to 5).
COHESION_K = 100.0
COHESION_POWER = 0.5

#: Grid points for that entropy. The kernel at ``k = 100`` is only 0.01 wide,
#: so it needs a finer grid than the default to give a steady answer.
COHESION_POINTS = points_for(COHESION_K)

#: Most pairs worked on at once.
_CHUNK = 4096


def pooled(rows: Array, weights: Array) -> Array:
    """Pooled distribution: the weighted mean of the rows (Incoherence paper, eq. 5)."""
    return weights @ rows


def incoherence(
    entropies: Array, pooled_entropy: float, max_entropy: float, weights: Array
) -> float:
    """Incoherence (Incoherence paper, eq. 9).

    Trials that agree pool into the distribution they already share, so the
    pooled entropy equals the entropy of each trial. Trials that disagree pool
    into something more spread out than any one of them. Incoherence is the
    weighted size of that gap, scaled to run from 0 to 1.

    Parameters
    ----------
    entropies
        Entropy of each trial.
    pooled_entropy
        Entropy of the pooled distribution.
    max_entropy
        The largest entropy the estimator can return.
    weights
        One weight per trial, summing to 1.
    """
    if pooled_entropy == 0:
        return 0.0
    spread = weights @ (pooled_entropy - entropies) ** 2
    return float(np.clip(np.sqrt(spread / (pooled_entropy * max_entropy)), 0.0, 1.0))


def pairwise_divergences(rows: Array, entropy: Entropy, weights: Array) -> Array:
    """Jensen-Shannon divergence between every pair of trials (Cohesion paper, eq. 2).

    For each pair, this is the entropy of the two trials pooled, less the
    weighted mean of their own entropies.

    Returns
    -------
    Array
        One value in [0, 1] per pair, in the order of ``numpy.triu_indices``.
        This is the condensed form that ``scipy.spatial.distance.squareform`` reads.
    """
    a, b = np.triu_indices(len(rows), k=1)
    wa = weights[a] / (weights[a] + weights[b])
    wb = 1 - wa
    entropies = entropy(rows)
    out = np.empty(len(a))
    step = max(1, min(_CHUNK, _BLOCK // rows.shape[1]))
    for start in range(0, len(a), step):
        s = slice(start, start + step)
        pair = wa[s, None] * rows[a[s]] + wb[s, None] * rows[b[s]]
        out[s] = entropy(pair) - wa[s] * entropies[a[s]] - wb[s] * entropies[b[s]]
    return np.clip(out, 0.0, 1.0)


def cohesion(divergences: Array, points: int = COHESION_POINTS) -> float:
    """Cohesion from the pairwise divergences (Cohesion paper, eq. 3 to 5).

    Divergences that bunch around a few values mean the trials fall into a few
    groups, which gives low entropy and high cohesion.

    Parameters
    ----------
    divergences
        One value in [0, 1] per pair of trials.
    points
        The number of grid points for the entropy of the divergences.
    """
    density = densities(divergences, COHESION_K, points)
    entropy = density_entropy(density, COHESION_K, COHESION_POWER)
    return float((1 - entropy) ** 2)
