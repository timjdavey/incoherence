import numpy as np

from .bins import binspace, binint
from .discrete import Discrete
from .continuous import Continuous


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


class Correlation(Continuous):
    """
    Is a wrapper function around Continuous.
    Where instead of passing observations, you pass the 
    x, y numeric values and it will automatically create ensembles for your.
    So that you can easily use it as a correlation metric.
    
    :x: list of x data points
    :y: list of y data points
    :ensemble_count: None, how many ensembles to use instead of a blend
    """   
    def __init__(self, x, y, ensemble_count=None, *args, **kwargs):
        self.x = np.array(x) # ensembles
        self.y = np.array(y) # to be binned or continuous
        
        # needs to be odd, otherwise symmetrical distributions always come out lowest
        minimum = 3
        blend_maximum = int(np.log(len(self.x)))
        obs = []
        labels = []
    
        # where you create progressively more ensembles
        # out of the data, but just add them all for
        # comparison
        # since they're different sizes
        # the defaulting weighting system
        # will count for the various proportioning
        if ensemble_count is None:
            for e in binint(minimum, blend_maximum):
                obs_i, labels_i = digitize(self.x, self.y, e)
                # need to add individually, as ragged lengths
                for row in obs_i:
                    obs.append(row)
                for label in labels_i:
                    labels.append(label)
        else:
            obs, labels = digitize(self.x, self.y, ensemble_count)
        
        super().__init__(obs, labels=labels,\
                metrics=('incoherence', ), *args, **kwargs)
            

    def correlations(self):
        """ Returns the standard correlation metrics for reference """
        from scipy.stats import pearsonr, spearmanr, kendalltau
        corrs = {}
        for name, func in (('pearson', pearsonr),
            ('spearman', spearmanr),
            ('kendall', kendalltau)):
            val, p = func(self.x, self.y)
            corrs[name] = val
            corrs['%s_p' % name] = p
        
        corrs['incoherence'] = self.incoherence
        return corrs


