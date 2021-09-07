import numpy as np
from functools import cached_property

from .bins import binr
from .ergodic import ErgodicEnsemble


def bin_suggestion(total, ratio=20, ensembles=(5,10), bins=(3,10)):
    """
    Creates a reasonable suggested number of default bins.
    Where ideally we have at least `ratio` (i.e. 20) observations per bin grid (i.e. ensemble*bin).
    So to enable the the "at least" mode, we give a maximum number of ensembles + bins (10 & 10).
    But give reasonable base level of ensembles & bins (5 & 3).
    """
    suggested = (total/ratio)**0.5
    ensemble_count = min(max(suggested, ensembles[0]), ensembles[1])
    bin_count = min(max(suggested, bins[0]), bins[1])
    return int(ensemble_count), int(bin_count)


def cont_to_disc(X, Y, count):
    """
    Continous to Discrete
    
    Takes a set of continuous 2D data of `X` and `Y` numeric values
    Returns the data grouped into `count` equal ensembles
    
    :X: list of numerics
    :Y: list of numerics
    :count: defaults to having on average 20 observations per ensemble
    """
    if len(X) != len(Y):
        raise IndexError("Length of X & Y must match %s != %s" % (len(X), len(Y)))
    
    ensembles = binr(X.min(), X.max(), int(count))
    
    # group using a dict
    obs = dict([(str(b), []) for b in ensembles[:-1]])
    for xi, x in enumerate(X):
        for bi, b in enumerate(ensembles):
            if x >= b and x <= ensembles[bi+1]:
                try:
                    obs[str(b)].append(Y[xi])
                    break
                except KeyError:
                    raise KeyError("Try reset_index on the filtered pandas dataframe")
    
    # returns observations as a dict
    return list(obs.values()), list(obs.keys())


class ErgodicCorrelation(ErgodicEnsemble):
    """
    Is a wrapper class around ErgodicEnsemble.
    Where instead of passing observations, you pass the 
    x, y numeric values and it will automatically create ensembles for your.
    So that you can easily use it as a correlation metric.
    
    inputs
    :counts: (ensemble_count, bin_count) tuple(ints) of counts

    functions
    :metrics: results a dict of common correlation metrics
    """    
    def __init__(self, x, y, counts=None, *args, **kwargs):
        self.x = np.array(x)
        self.y = np.array(y)
                
        # create sensible bins
        if counts is None:
            ensemble_count, bin_count = bin_suggestion(len(self.x))
        else:
            ensemble_count, bin_count = counts
        
        # turn the continous data into discrete ensembles
        obs, labels = cont_to_disc(self.x, self.y, ensemble_count)
        bins = binr(self.y.min(), self.y.max(), bin_count)
        
        # create an ErgodicEnsemble standard
        super().__init__(obs, bins, labels=labels, *args, **kwargs)
    
    @cached_property
    def correlations(self):
        from scipy.stats import pearsonr, spearmanr, kendalltau
        return {
            "pearson": pearsonr(self.x, self.y)[0],
            "spearman": spearmanr(self.x, self.y)[0],
            "kendall": kendalltau(self.x, self.y)[0],
            "complexity": self.complexity,
        }