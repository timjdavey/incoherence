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
        divs = [(ergodic_entropy - e)**2 for e in entropies]
        return (np.average(divs, weights=weights) / ergodic_entropy)**0.5


TAU_BOOST = 350

def tau2(comp, states, boost=None):
    """
    Calculates the ergodic complexity measures, transformed for use with
    a Chi-Square distribution.

    :returns: tau2 measure and tau-p as tuple of floats
    """
    import scipy as sp
    if boost is None: boost = TAU_BOOST
    tau = (comp**2)*states*boost
    tau_p = 1 - sp.stats.chi2.cdf(tau, states)
    return tau, tau_p


LEGEND = {
    'ensemble': ('Mean esemble entropy','orange'),
    'ergodic': ('Ergodic entropy','firebrick'),
    'divergence': ('Erogodic divergence','forestgreen'),
    'complexity': ('Ergodic complexity','blueviolet'),
    'tau2': ('tau2 Conversion of ergodic complexity', 'gold'),
    'tau2p': ('tau2 p-value', 'cyan'),
    'entropies': ('Entropies of individual ensembles','crest'),
}


def measures(pmfs, weights=None, with_entropies=False, boost=None, **kwargs):
    """ Returns all metrics """
    ents = ensemble_entropies(pmfs, **kwargs)
    ensemble = np.mean(ents)
    ergodic = ergodic_entropy(pmfs, **kwargs)
    diver = divergence(ergodic, ents, weights)
    comp = complexity(ergodic, ents, weights)
    tau2p = tau2(comp, len(pmfs), boost)

    metrics = {
        'ensemble': ensemble,
        'ergodic': ergodic,
        'divergence': diver,
        'complexity': comp,
        'tau2': tau2p[0],
        'tau2p': tau2p[1],
    }
    if with_entropies:
        metrics['entropies'] = ents
    return metrics


