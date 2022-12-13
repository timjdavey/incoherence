import numpy as np
import scipy as sp
from itertools import permutations, combinations
from .entropy import ensemble_entropies, pooled_entropy
from .densityvar import density_variance

def js_divergence(p_entropy, entropies, weights, power=1):
    """ Jenson Shannon Divergence """
    divs = [(p_entropy - e)**power for e in entropies]
    return np.average(divs, weights=weights)


def radial_divergences(data, discrete=True, normalise=False, entropies=None):
    """
    Returns the JS divergences for each pair of data

    :data: if discrete is True, in the form of histograms,
        if continuous just all the observations normalised between (0,1)
    :discrete: True, is the data histograms or continuous observations
    :normalise: False, will automatically normalise the output
    :entropies: list, saves calculating the individual entropies for data if already done
    """
    divergences = []
    if entropies is None:
        if discrete: entropies = ensemble_entropies(data)
        else: entropies = [density_variance(a) for a in data]

    indices = list(range(len(data)))
    for a,b in combinations(indices, 2):
        if discrete:
            p_entropy = pooled_entropy([data[a],data[b]])
        else:
            p_entropy = density_variance(np.concatenate([data[a],data[b]]))

        div = js_divergence(p_entropy, (entropies[a], entropies[b]), None)
        divergences.append(div)

    divergences = np.array(divergences)

    if normalise:
        # if continuous will already be normalised to (0,1)
        if discrete:
            divergences /= np.log(len(data[0])-1)
    
        # deal with float errors
        divergences[divergences<0] = 0
        divergences[divergences>1] = 1

    return divergences


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