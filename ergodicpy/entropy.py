import numpy as np
from scipy.stats import entropy as shannon_entropy


def ensemble_entropies(pmfs, **kwargs):
    """
    Returns an array of entropies, given array of pmfs
    """
    return [shannon_entropy(p, **kwargs) for p in pmfs]


def observation_weights(hists, weights=None):
    """
    Default weight strategy of N_k/N

    :hists: histograms of each distribution
    :weights: can pass weights which if exist,
        returns those so can use this function as a default mechanism
    """
    if weights is None:
        # N_k/N default
        ws = np.array(hists).sum(axis=1)
        return ws/ws.sum()
    elif weights is False:
        # actively set it to turn off
        # then have them equal weight
        N = len(hists)
        return np.ones(N)/N
    else:
        # no need to normalize as np does that
        return weights


def point_pmf(pmfs, weights=None):
    """
    For a given array of pmfs
    Returns the weighted pointwise ergodic pmf
    """
    default_weights = observation_weights(pmfs, weights)
    normed = np.array([row/row.sum() for row in np.array(pmfs)])
    return np.average(normed, weights=default_weights, axis=0)


def ergodic_entropy(pmfs, weights=None, **kwargs):
    """
    Returns the entropy of the ergodic ensemble
    """
    return shannon_entropy(point_pmf(pmfs, weights), **kwargs)

