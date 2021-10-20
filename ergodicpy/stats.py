import numpy as np
import scipy as sp
from .entropy import ensemble_entropies, observation_weights, ergodic_entropy


def js_divergence(ergodic_entropy, entropies, weights, power=1):
    """ Shannon Jenson Divergence """
    divs = [(ergodic_entropy - e)**power for e in entropies]
    return np.average(divs, weights=weights)


def complexity(ergodic_entropy, entropies, weights):
    """
    Ergodic complexity calculation

    :ergodic_entropy: can use `ergodic_entropy()` to calc
    :entropies: can use `entropies()` to calc
    """
    if ergodic_entropy == 0:
        return 0.0
    else:
        divs = js_divergence(ergodic_entropy, entropies, weights, power=2)
        return (divs / ergodic_entropy)**0.5


def kl_divergences(references, observed, power=1):
    """
    Returns an array of KL divergences
    For a given array of histogram `references`
    and an `observed` histogram.
    """
    return np.array([sp.stats.entropy(observed, h)**power for h in references])


LEGEND = {
    'ensemble': ('Mean esemble entropy','orange'),
    'ergodic': ('Ergodic entropy','firebrick'),
    'divergence': ('Erogodic divergence','forestgreen'),
    'complexity': ('Ergodic complexity','blueviolet'),
    'is_complex': ('Ergodic complexity greater than threshold', 'cyan'),
    'entropies': ('Entropies of individual ensembles','crest'),
    'weights': ('Weights of each ensemble', 'red'),
}

THRESHOLD = 0.07

def measures(pmfs, weights=None, with_meta=False, threshold=THRESHOLD, **kwargs):
    """ Returns all metrics """
    ents = ensemble_entropies(pmfs, **kwargs)
    weights = observation_weights(pmfs, weights)
    ergodic = ergodic_entropy(pmfs, weights, **kwargs)
    comp = complexity(ergodic, ents, weights)

    metrics = {
        'ensemble': np.average(ents, weights=weights),
        'ergodic': ergodic,
        'divergence': js_divergence(ergodic, ents, weights),
        'complexity': comp,
        'distance': comp**0.5,
        'is_complex': 1 if comp > threshold else 0,
    }
    if with_meta:
        metrics['entropies'] = ents
        metrics['weights'] = weights
    return metrics
