import numpy as np

def shannon_entropy(pmf, normalise=False, unit='bits'):
    """
    Calculates the Shannon entropy for a
    discrete probability mass function.

    Discards all zero probabilty states.
    Performs various checks.

    :normalise: _False_. Will automatically normalize the pmf for you.
    :units: _'bits'_ (or _'nats'_). The unit of entropy to be returned.
    """
    pmf = np.array(pmf)

    if pmf.size < 2:
        raise ValueError('len(pmf) is < 2 %s' % pmf)
    
    # setting the base
    if unit == 'bits':
        log = np.log2
    elif unit == 'nats':
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


def ensemble_entropy(pmfs, normalise=True, units='bits'):
    """
    For a given array of pmfs
    Returns the ensemble entropy (the mean of the entropy for pmfs)
    """
    return np.mean([shannon_entropy(p, normalise, units) for p in pmfs])


def ergodic_entropy(pmfs, normalise=True, units='bits'):
    """
    For a given array of pmfs
    Returns the ergodic pmf, i.e. the mean
    """
    return shannon_entropy(np.mean(pmfs, axis=0), normalise, units)


def complexity_from_averages(avg_ensemble_entropy, ergodic_entropy):
    """ Basic function for handling zero errors """
    # handle zero division error
    # since erg should be > ensemble always
    # checking for erg is 0 means ensemble is zero too 
    if ergodic_entropy == 0 or avg_ensemble_entropy > ergodic_entropy:
        return 0.0
    else:
        return 1 - (avg_ensemble_entropy / ergodic_entropy)

def measures(pmfs, normalise=True, units='bits'):
    """ Returns all metrics """
    ensemble = ensemble_entropy(pmfs, normalise, units)
    ergodic = ergodic_entropy(pmfs, normalise, units)
    divergence = ergodic - ensemble
    complexity = complexity_from_averages(ensemble, ergodic)
    return ensemble, ergodic, divergence, complexity


