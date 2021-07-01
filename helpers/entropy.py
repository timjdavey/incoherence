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
    if ergodic_entropy == 0 or avg_ensemble_entropy > ergodic_entropy:
        return 0.0
    else:
        return 1 - (avg_ensemble_entropy / ergodic_entropy)


def sigmoid_complexity(complexity, a=4, b=-12,k=1.04,l=0.03):
    """
    Similar function to the basic entropy calculation above,
    except it amplifies the limits by passing it through
    a sigmoid function.

    a & b are defaulted so that it's skewed to present
    complexities > 0.3 as more prominent and > 0.5 as basically 1.0

    You can visualise the variables below
    https://www.desmos.com/calculator/hcs6p0jhfn

    In some cases it comes off worse, but the big benefit
    of this is it makes the results more consistent across
    - what bin strategies you choose
    - how fine grained your ensembles are
    - how many obversations you have
    """
    calc = k/(1+np.exp(a+b*complexity)) - l
    # make sure only returns in bounds [0.0, 1.0]
    return min(max(calc, 0.0), 1.0)



def int_entropy(observations):
    """
    Work out entropy for a given set of observations
    Using a bin strategy of just int bounds
    """
    observations = np.array(observations)
    bins = np.arange(observations.max()+2)
    pmf, nbins = np.histogram(observations, bins=bins)
    return shannon_entropy(pmf, True)

