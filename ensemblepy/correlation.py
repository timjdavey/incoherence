import numpy as np

from .bins import binspace, binint
from .base import Ensembles

def digitize(X, Y, count):
    """
    Continous to Discrete
    
    Takes a set of continuous 2D data of `X` and `Y` numeric values
    Returns the data grouped into `count` equal ensembles
    
    :X: list of numerics
    :Y: list of numerics
    :count: defaults to having on average 20 observations per ensemble
    """
    X = np.array(X)
    Y = np.array(Y)

    if len(X) != len(Y):
        raise IndexError("Length of X & Y must match %s != %s" % (len(X), len(Y)))
    
    ensembles = binspace(X.min(), X.max(), int(count))
    
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


class Correlation(Ensembles):
    """
    Is a wrapper class around Ensembles.
    Where instead of passing observations, you pass the 
    x, y numeric values and it will automatically create ensembles for your.
    So that you can easily use it as a correlation metric.
    
    inputs
    :counts: (ensemble_count, bin_count) tuple(ints) of counts

    functions
    :metrics: results a dict of common correlation metrics
    """    
    def __init__(self, x, y, ensembles=None, *args, **kwargs):
        self.x = np.array(x)
        self.y = np.array(y)
        
        # create a blended set of ensembles
        if ensembles is None:
            maximum = int(np.log(len(self.x)))
            minimum = 3
            obs = []
            labels = []
            # where you create progressively more ensembles
            # out of the data, but just add them all for
            # comparison
            # since they're different sizes
            # the defaulting weighting system
            # will count for the various proportioning
            # into the ergodic ensemble & final calc
            for e in binint(minimum, maximum):
                obs_i, labels_i = digitize(self.x, self.y, e)
                # need to add individually, as ragged lengths
                for row in obs_i:
                    obs.append(row)
                for label in labels_i:
                    labels.append(label)
        else:
            # turn the continous data into discrete ensembles
            obs, labels = digitize(self.x, self.y, ensembles)
        
        
        # create an Ensembles standard
        super().__init__(obs, labels=labels, *args, **kwargs)

    
    @property
    def correlations(self):
        from scipy.stats import pearsonr, spearmanr, kendalltau
        pr, pp = pearsonr(self.x, self.y)
        sr, sp = spearmanr(self.x, self.y)
        kr, kp = kendalltau(self.x, self.y)
        return {
            "pearson": pr,
            "pearson_p": pp,
            "spearman": sr,
            "spearman_p": sp,
            "kendall": kr,
            "kendall_p": kp,
            "incoherence": self.incoherence,
        }