import numpy as np
from .entropy import ensemble_entropies, observation_weights, ergodic_entropy


def divergence(ergodic_entropy, entropies, weights):
    """ Shannon Jenson Divergence """
    divs = [ergodic_entropy - e for e in entropies]
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
        divs = [(ergodic_entropy - e)**2 for e in entropies]
        return (np.average(divs, weights=weights) / ergodic_entropy)**0.5


TAU_BOOST = 280

def tau2(comp, states, boost=None):
    """
    Calculates the ergodic complexity measures, transformed for use with
    a Chi-Square distribution.

    :returns: tau2 measure and tau-p as tuple of floats
    """
    try:
        import scipy as sp
    except ImportError:
        return None, None

    if boost is None: boost = TAU_BOOST
    tau = (comp**2)*np.log(states)*boost
    tau_p = 1 - sp.stats.chi2.cdf(tau, 1)
    return tau, tau_p


LEGEND = {
    'ensemble': ('Mean esemble entropy','orange'),
    'ergodic': ('Ergodic entropy','firebrick'),
    'divergence': ('Erogodic divergence','forestgreen'),
    'complexity': ('Ergodic complexity','blueviolet'),
    'weights': ('Weights of each ensemble', 'red'),
    'tau2': ('tau2 Conversion of ergodic complexity', 'gold'),
    'tau2p': ('tau2 p-value', 'cyan'),
    'entropies': ('Entropies of individual ensembles','crest'),
}


def measures(pmfs, weights=None, with_meta=False, tau_boost=None, **kwargs):
    """ Returns all metrics """
    ents = ensemble_entropies(pmfs, **kwargs)
    weights = observation_weights(pmfs, weights)
    ensemble = np.average(ents, weights=weights)
    ergodic = ergodic_entropy(pmfs, weights, **kwargs)
    diver = divergence(ergodic, ents, weights)
    comp = complexity(ergodic, ents, weights)
    tau2p = tau2(comp, len(pmfs[0]), tau_boost)

    metrics = {
        'ensemble': ensemble,
        'ergodic': ergodic,
        'divergence': diver,
        'complexity': comp,
        'tau2': tau2p[0],
        'tau2p': tau2p[1],
    }
    if with_meta:
        metrics['entropies'] = ents
        metrics['weights'] = weights
    return metrics
