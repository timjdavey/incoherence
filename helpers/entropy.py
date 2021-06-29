import numpy as np

def shannon_entropy(pmf, normalise=False):
    """
    Calculates the Shannon entropy for a
    discrete probability mass function.

    Discards all zero probabilty states.
    """
    pmf = np.array(pmf)

    if normalise:
        pmf = pmf/pmf.sum()
    
    # still need to run check for empty pdfs
    # assert_almost to deal with float point errors
    try:
        np.testing.assert_almost_equal(pmf.sum(), 1.0)
    except AssertionError:
        raise ValueError("pmf %s=%s is not normalised" % (pmf,pmf.sum()))

    # discard anything of zero probability
    pmf = np.ma.masked_equal(pmf,0).compressed()
    # could use nansum below, but rightly gives runtime warnings

    # add 0 as workaround to avoid -0.0
    return -np.sum(pmf * np.log2(pmf)) + 0



def complexity(avg_ensemble_entropy, ergodic_entropy):
    """ This function might change, which is why it's encapsulated """
    # handle zero division error
    # since erg should be > ensemble always
    # checking for erg is 0 means ensemble is zero too 
    if ergodic_entropy == 0:
        return 0
    else:
        return 1 - (avg_ensemble_entropy / ergodic_entropy)



def int_entropy(observations):
    """
    Work out entropy for a given set of observations
    Using a bin strategy of just int bounds
    """
    observations = np.array(observations)
    bins = np.arange(observations.max()+2)
    pmf, nbins = np.histogram(observations, bins=bins)
    return shannon_entropy(pmf, True)

