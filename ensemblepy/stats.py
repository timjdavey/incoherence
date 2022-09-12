import numpy as np
import scipy as sp
from .entropy import ensemble_entropies, observation_weights, pooled_entropy


def js_divergence(p_entropy, entropies, weights, power=1):
    """ Shannon Jenson Divergence """
    divs = [(p_entropy - e)**power for e in entropies]
    return np.average(divs, weights=weights)


def complexity(p_entropy, entropies, weights):
    """
    Ensemble complexity calculation

    :p_entropy: can use `pooled_entropy()` to calc
    :entropies: can use `entropies()` to calc
    """
    if p_entropy == 0:
        return 0.0
    else:
        divs = js_divergence(p_entropy, entropies, weights, power=2)
        return (divs / p_entropy)**0.5


def kl_divergences(references, observed, power=1):
    """
    Returns an array of KL divergences
    For a given array of histogram `references`
    and an `observed` histogram.
    """
    return np.array([sp.stats.entropy(observed, h)**power for h in references])


LEGEND = {
    'ensemble': ('Mean ensemble entropy','orange'),
    'pooled': ('Pooled entropy','firebrick'),
    'divergence': ('Ensemble divergence','forestgreen'),
    'complexity': ('Ensemble complexity','blueviolet'),
    'entropies': ('Entropies of individual ensembles','crest'),
    'weights': ('Weights of each ensemble', 'red'),
}

THRESHOLD = 0.07

def measures(pmfs, weights=None, with_meta=False, **kwargs):
    """ Returns all metrics """
    ents = ensemble_entropies(pmfs, **kwargs)
    weights = observation_weights(pmfs, weights)
    pooled = pooled_entropy(pmfs, weights, **kwargs)
    comp = complexity(pooled, ents, weights)

    metrics = {
        'ensemble': np.average(ents, weights=weights),
        'pooled': pooled,
        'divergence': js_divergence(pooled, ents, weights),
        'complexity': comp,
    }
    if with_meta:
        metrics['entropies'] = ents
        metrics['weights'] = weights
    return metrics
