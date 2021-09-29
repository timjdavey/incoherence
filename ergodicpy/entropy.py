import numpy as np

DEFAULT_UNITS = 'nats'

def shannon_entropy(pmf, normalise=True, units=DEFAULT_UNITS):
    """
    Calculates the Shannon entropy for a
    discrete probability mass function.

    Discards all zero probabilty states.
    Performs various checks.

    :normalise: _False_. Will automatically normalize the pmf for you.
    :units: _'nats'_ (or _'bits'_). The unit of entropy to be returned.
    """
    pmf = np.array(pmf)

    if pmf.size < 2:
        raise ValueError('len(pmf) is < 2 %s' % pmf)
    
    # setting the base
    if units == 'bits':
        log = np.log2
    elif units == 'nats' or units is None: # nats is default
        log = np.log
    else:
        raise ValueError('Please specify a `unit` of `bits` or `nats`')

    if normalise:
        pmf = pmf/pmf.sum()
    else:
        try:
            # assert_almost to deal with float point errors
            np.testing.assert_almost_equal(pmf.sum(), 1.0)
        except AssertionError:
            raise ValueError("pmf %s=%s is not normalised" % (pmf,pmf.sum()))

    # discard anything of zero probability
    pmf = np.ma.masked_equal(pmf,0).compressed()
    # could use nansum below, but rightly gives runtime warnings

    # add 0 as workaround to avoid -0.0
    entropy = -np.sum(pmf * log(pmf)) + 0

    return entropy


def entropy_from_obs(observations, bins, **kwargs):
    """ Simple function wrap np.histogram """
    hist, _ = np.histogram(observations, bins=bins)
    return shannon_entropy(hist, **kwargs)


def ensemble_entropies(pmfs, **kwargs):
    """ Returns an array of entropies, given array of pmfs"""
    return [shannon_entropy(p, **kwargs) for p in pmfs]


def ergodic_entropy(pmfs, **kwargs):
    """
    For a given array of pmfs
    Returns the ergodic pmf, i.e. the mean
    """
    return shannon_entropy(np.sum(pmfs, axis=0), **kwargs)


def divergence(ergodic_entropy, entropies, weights):
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
#        divs = [(1 - e/ergodic_entropy)**2 for e in entropies]
#        return (np.average(divs, weights=weights))**0.5

        divs = [(ergodic_entropy - e)**2 for e in entropies]
        return (np.average(divs, weights=weights) / ergodic_entropy)**0.5


LEGEND = {
    'ensemble': ('Mean esemble entropy','orange'),
    'ergodic': ('Ergodic entropy','firebrick'),
    'divergence': ('Erogodic divergence','forestgreen'),
    'complexity': ('Ergodic complexity','blueviolet'),
    'entropies': ('Entropies of individual ensembles','crest'),
}


def measures(pmfs, weights=None, with_entropies=False, **kwargs):
    """ Returns all metrics """
    ents = ensemble_entropies(pmfs, **kwargs)
    ensemble = np.mean(ents)
    ergodic = ergodic_entropy(pmfs, **kwargs)

    metrics = {
        'ensemble': ensemble,
        'ergodic': ergodic,
        'divergence': divergence(ergodic, ents, weights),
        'complexity': complexity(ergodic, ents, weights),
    }
    if with_entropies:
        metrics['entropies'] = ents
    return metrics


