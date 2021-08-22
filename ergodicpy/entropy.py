import numpy as np

def shannon_entropy(pmf, normalise=False, units=None):
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
    return -np.sum(pmf * log(pmf)) + 0


def entropy_from_obs(observations, bins, units=None):
    """ Simple function wrap np.histogram """
    hist, _ = np.histogram(observations, bins=bins)
    return shannon_entropy(hist, True, units)


def ensemble_entropies(pmfs, normalise=True, units=None):
    """ Returns an array of entropies, given array of pmfs"""
    return [shannon_entropy(p, normalise, units) for p in pmfs]


def ensemble_entropy(pmfs, normalise=True, units=None):
    """
    For a given array of pmfs
    Returns the ensemble entropy (the mean of the entropy for pmfs)
    """
    return np.mean(ensemble_entropies(pmfs, normalise, units))


def ergodic_entropy(pmfs, normalise=True, units=None):
    """
    For a given array of pmfs
    Returns the ergodic pmf, i.e. the mean
    """
    return shannon_entropy(np.mean(pmfs, axis=0), normalise, units)

def zero_divide(numerator, demoniator):
    if demoniator == 0:
        return 0.0
    else:
        return numerator/demoniator

def complexity(ergodic_entropy, entropies):
    """
    Ergodic complexity calculation

    :ergodic_entropy: can use `ergodic_entropy()` to calc
    :entropies: can use `entropies()` to calc

    Complexity is unitless, however, the entropies passed must all be of the same unit
    """
    if ergodic_entropy == 0:
        return 0.0
    else:
        divs = [(ergodic_entropy - e)**2 for e in entropies]
        return (np.mean(divs) / ergodic_entropy)

LEGEND = {
    'ensemble': ('Ensemble entropy','orange'),
    'ergodic': ('Ergodic entropy','firebrick'),
    'divergence': ('Erogodic divergence','forestgreen'),
    'complexity': ('Ergodic complexity','blueviolet'),
    'entropies': ('Entropies of individual ensembles','crest'),
}


def measures(pmfs, normalise=True, units=None, with_entropies=False):
    """ Returns all metrics """
    ents = ensemble_entropies(pmfs, normalise, units)
    ensemble = np.mean(ents)
    ergodic = ergodic_entropy(pmfs, normalise, units)
    comp = complexity(ergodic, ents)

    metrics = {
        'ensemble': ensemble,
        'ergodic': ergodic,
        'divergence': ergodic - ensemble,
        'complexity': comp,
    }
    if with_entropies:
        metrics['entropies'] = ents
    return metrics


