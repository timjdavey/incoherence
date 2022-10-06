import numpy as np
import scipy as sp
from itertools import permutations
from .entropy import ensemble_entropies, observation_weights, pooled_entropy


def js_divergence(p_entropy, entropies, weights, power=1):
    """ Jenson Shannon Divergence """
    divs = [(p_entropy - e)**power for e in entropies]
    return np.average(divs, weights=weights)


def incoherence(p_entropy, entropies, weights=None):
    """
    incoherence calculation

    :p_entropy: can use `pooled_entropy()` to calc
    :entropies: can use `entropies()` to calc
    :weights: _None_
    """
    if weights is None:
        weights = np.ones(len(entropies))
    if p_entropy == 0:
        return 0.0
    else:
        divs = js_divergence(p_entropy, entropies, weights, power=2)
        return (divs / p_entropy)**0.5


def kl_divergences(references, observed=None, power=1, func=sp.stats.entropy):
    """
    Returns an array of KL divergences
    For a given array of histogram `references`
    and an `observed` histogram.
    If `observed` is default None, compared against the permutation of references.
    """
    if observed is None:
        return np.array([func(a, b) for a, b in permutations(references, 2)])
    else:
        return np.array([func(observed, h)**power for h in references])


LEGEND = {
    'ensemble': ('Mean entropy of individuals','orange'),
    'pooled': ('Entropy of pooled','firebrick'),
    'divergence': ('Divergence','forestgreen'),
    'incoherence': ('Incoherence','blueviolet'),
    'entropies': ('Entropies of individual ensembles','crest'),
    'weights': ('Weights of each ensemble', 'red'),
}

def measures(pmfs, weights=None, with_meta=False, **kwargs):
    """ Returns all metrics """
    ents = ensemble_entropies(pmfs, **kwargs)
    weights = observation_weights(pmfs, weights)
    pooled = pooled_entropy(pmfs, weights, **kwargs)

    metrics = {
        'ensemble': np.average(ents, weights=weights),
        'pooled': pooled,
        'divergence': js_divergence(pooled, ents, weights),
        'incoherence': incoherence(pooled, ents, weights),
    }
    if with_meta:
        metrics['entropies'] = ents
        metrics['weights'] = weights
    return metrics
