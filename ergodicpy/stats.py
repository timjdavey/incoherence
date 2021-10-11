import numpy as np
import scipy as sp
from .entropy import ensemble_entropies, observation_weights, ergodic_entropy


def divergence(ergodic_entropy, entropies, weights):
    """ Shannon Jenson Divergence """
    divs = [ergodic_entropy - e for e in entropies]
    return np.average(divs, weights=weights)


def complexity(ergodic_entropy, entropies, weights, alt=False):
    """
    Ergodic complexity calculation

    :ergodic_entropy: can use `ergodic_entropy()` to calc
    :entropies: can use `entropies()` to calc
    """
    if ergodic_entropy == 0:
        return 0.0
    else:
        if alt:
            divs = [(1 - e/ergodic_entropy)**2 for e in entropies]
            return np.average(divs, weights=weights)**0.5
        else:
            divs = [(ergodic_entropy - e)**2 for e in entropies]
            return (np.average(divs, weights=weights) / ergodic_entropy)**0.5            


LEGEND = {
    'ensemble': ('Mean esemble entropy','orange'),
    'ergodic': ('Ergodic entropy','firebrick'),
    'divergence': ('Erogodic divergence','forestgreen'),
    'complexity': ('Ergodic complexity','blueviolet'),
    'c2': ('tau2 Conversion of ergodic complexity', 'gold'),
    'alt': ('tau2 p-value', 'cyan'),
    'alt2': ('tau2 p-value', 'cyan'),
    'entropies': ('Entropies of individual ensembles','crest'),
    'weights': ('Weights of each ensemble', 'red'),
}


def measures(pmfs, weights=None, with_meta=False, **kwargs):
    """ Returns all metrics """
    ents = ensemble_entropies(pmfs, **kwargs)
    weights = observation_weights(pmfs, weights)
    ensemble = np.average(ents, weights=weights)
    ergodic = ergodic_entropy(pmfs, weights, **kwargs)
    c = complexity(ergodic, ents, weights)
    a = complexity(ergodic, ents, weights, alt=True)

    metrics = {
        'ensemble': ensemble,
        'ergodic': ergodic,
        'divergence': divergence(ergodic, ents, weights),
        'complexity': c,
        'c2': c**2,
        'alt': a,
        'alt2': a**2,
    }
    if with_meta:
        metrics['entropies'] = ents
        metrics['weights'] = weights
    return metrics
