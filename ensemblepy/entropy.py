import numpy as np
from scipy.stats import entropy as shannon_entropy


def ensemble_entropies(pmfs, **kwargs):
    """
    Returns an array of entropies, given array of pmfs
    """
    return [shannon_entropy(p, **kwargs) for p in pmfs]


def get_weights(data, weights=None, discrete=True):
    """
    Default weight strategy of N_k/N

    :data: histograms of each distribution or set of observations if continuous
    :weights: _None_, can pass weights which if exist,
        returns those so can use this function as a default mechanism
        set as _False_ if want to ignore weights
    :discrete: _True_, if _False_ specify data is continuous
        and use observation count instead
    """
    if weights is None:
        if discrete:
            # N_k/N default
            ws = np.array(data).sum(axis=1)
        else:
            # continuous
            ws = np.array([len(h) for h in data])
        return ws / ws.sum()
    elif weights is False:
        # actively set it to turn off
        # then have them equal weight
        N = len(data)
        return np.ones(N) / N
    else:
        # no need to normalize as np does that
        return weights


def point_pmf(pmfs, weights=None):
    """
    For a given array of pmfs
    Returns the weighted pointwise pooled pmf
    """
    default_weights = get_weights(pmfs, weights)
    normed = np.array([row / row.sum() for row in np.array(pmfs)])
    return np.average(normed, weights=default_weights, axis=0)


def pooled_entropy(pmfs, weights=None, **kwargs):
    """
    Returns the entropy of the pooled ensemble
    """
    return shannon_entropy(point_pmf(pmfs, weights), **kwargs)
