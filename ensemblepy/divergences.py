import numpy as np
import scipy as sp
from itertools import permutations, combinations
from .entropy import ensemble_entropies, pooled_entropy
from .densityvar import density_variance

def js_divergence(p_entropy, entropies, weights, power=1):
    """ Jenson Shannon Divergence """
    divs = [(p_entropy - e)**power for e in entropies]
    return np.average(divs, weights=weights)


def radial_divergences(data, discrete=True):
    """
    Returns the JS divergences for each pair of data

    :data: if discrete is True, in the form of histograms,
        if continuous just all the observations normalised between (0,1)
    :discrete: True, is the data histograms or continuous observations
    """
    divergences = []
    for a,b in combinations(data, 2):
        if discrete:
            p_entropy = pooled_entropy([a,b])
            entropies = ensemble_entropies([a,b])
        else:
            p_entropy = density_variance(np.concatenate([a,b]))
            entropies = [
                density_variance(a),
                density_variance(b)
            ]
        div = ep.js_divergence(p_entropy, entropies, None)
        divergences.append(div)
    return np.array(divergences)


def kl_divergences(data, compare=None):
    """
    Returns the KL divergences for each pair of `data`.
    Unless an `compare` set is provided, in which case combined against that.

    :data: core reference data
    :compare: None, optional data to compare against
    """
    if compare is None:
        return np.array([sp.stats.entropy(a, b) for a, b in combinations(data, 2)])
    else:
        return np.array([sp.stats.entropy(compare, h) for h in data])