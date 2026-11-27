"""The equations from the papers, as pure functions.

Every function works on rows and an entropy estimator, so the same code serves
discrete and continuous data. See ``_entropy`` for what a row is.
"""

from collections.abc import Callable

import numpy as np

from ._entropy import Array, densities, density_entropy, points_for

Entropy = Callable[[Array], Array]

#: Power for the entropy of the pairwise divergences (Cohesion paper, eq. 3 to 5).
COHESION_POWER = 0.5

#: How much that entropy's ``k`` grows for each trial. See ``experiments/k``.
COHESION_RATE = 8.0

#: The fixed ``k`` the papers used for that entropy, before it grew with trials.
PAPER_COHESION_K = 100.0

#: Most pairs worked on at once.
_CHUNK = 4096

#: Largest temporary array, in values, built while working through the pairs.
_BLOCK = 2**21


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

    Parameters
    ----------
    rows
        One distribution per trial.
    entropy
        The estimator that maps rows to one entropy each.
    weights
        One weight per trial. Each pair uses its two weights, rescaled to sum to 1.

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


def pair_weights(weights: Array) -> Array | None:
    """How much each pair of trials counts towards cohesion.

    A pair counts in proportion to the weights of both its trials, so a trial
    with almost no weight has almost no effect.

    Returns
    -------
    Array or None
        One value per pair, in the order of ``numpy.triu_indices``. ``None``
        when every trial has the same weight, so every pair counts the same.
    """
    if np.ptp(weights) == 0:
        return None
    a, b = np.triu_indices(len(weights), k=1)
    scaled = weights / weights.max()
    return scaled[a] * scaled[b]


def effective_trials(weights: Array) -> float:
    """How many trials the weights amount to, for setting cohesion's ``k``.

    This is the number of trials when every weight is the same. Otherwise it
    is ``1 / sum(weights ** 2)``, so a trial with almost no weight barely
    counts, just as it barely counts towards cohesion itself.

    Parameters
    ----------
    weights
        One weight per trial, summing to 1.
    """
    if np.ptp(weights) == 0:
        return float(len(weights))
    return float(1 / (weights @ weights))


def cohesion_k(trials: float) -> float:
    """Return the default ``k`` for the entropy of the divergences, 8 for each trial.

    More trials give more pairs, and more pairs need a finer kernel to tell
    a few groups from noise. The experiments in ``experiments/k`` found that
    ``8n`` reads noise lowest of every rule they tried, and beats the fixed
    100 the papers used.
    """
    return COHESION_RATE * trials


def cohesion(
    divergences: Array, k: float, points: int | None = None, weights: Array | None = None
) -> float:
    """Cohesion from the pairwise divergences (Cohesion paper, eq. 3 to 5).

    Divergences that bunch around a few values mean the trials fall into a few
    groups, which gives low entropy and high cohesion.

    Parameters
    ----------
    divergences
        One value in [0, 1] per pair of trials.
    k
        How quickly each divergence's influence falls away, from :func:`cohesion_k`
        by default.
    points
        The number of grid points for the entropy of the divergences. Leave as
        ``None`` for the number that ``k`` needs, see :func:`points_for`.
    weights
        How much each pair counts, from :func:`pair_weights`. Leave as ``None``
        to count every pair the same.
    """
    density = densities(divergences, k, points_for(k) if points is None else points, weights)
    entropy = density_entropy(density, k, COHESION_POWER)
    return float((1 - entropy) ** 2)
