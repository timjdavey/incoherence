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


def ensemble_entropies(pmfs, **kwargs):
    """ Returns an array of entropies, given array of pmfs"""
    return [shannon_entropy(p, **kwargs) for p in pmfs]


def ergodic_entropy(pmfs, weights, **kwargs):
    """
    For a given array of pmfs
    Returns the ergodic pmf, i.e. the mean
    """
    return shannon_entropy(np.average(pmfs, weights=weights, axis=0), **kwargs)


def observation_weights(hists, weights=None):
    """
    Default weight strategy of N_k/N

    :hists: histograms of each distribution
    :weights: can pass weights which if exist,
        returns those so can use this function as a default mechanism
    """
    if weights is None:
        # N_k/N default
        counts = np.array([np.sum(row) for row in hists])
        return counts/counts.sum()
    elif weights is False:
        # actively set it to turn off
        # then have them equal weight
        return np.ones(len(hists))/len(hists)
    else:
        # if specified
        # make sure they're normalized
        weights = np.array(weights)
        return weights/weights.sum()
        



