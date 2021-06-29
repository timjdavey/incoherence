import numpy as np

from .entropy import shannon_entropy, complexity


class ErgodicEnsemble:
    """
    A simple model to help calculate the 

    Contains some simple performance boosts, but also stores
    some helpful data to make visualisation simpler
    (hence why it's a class rather than just a function).
    
    intialization
    :observations: either takes a list or dict of the observations grouped by ensemble
    e.g. [[0,0,1,0], [0,1,0]] or {'UK':[0,0,1,0], 'US':[0,1,0]}
    if pass a dict, the keys will be used as a legend in the ensemble plots
    
    :bins: the bins to be used for the data e.g. np.linspace(data.min(), data.max(), 20)
    :ensemble_name: the name of the ensemble to be used plots

    properties
    :ensemble: the average ensemble entropy
    :ergodic: the entropy of the ergodic distribution
    :complexity: the ergodic complexity

    functions
    :plot: plots the ensemble & ergodic histograms
    :ridge: plots a ridge plot of the ensemble histograms
    :stats: prints all the stats in an easy to read format
    """
    def __init__(self, observations, bins, ensemble_name=None):

        if isinstance(observations, (list, np.ndarray)):
            self.observations = observations
            self.labels = None
            self.raw = None
        elif isinstance(observations, dict):
            self.observations = list(observations.values())
            self.labels = observations.keys()
            self.raw = observations
        else:
            raise TypeError(
                "observations is of type %s not list or dict" % type(observations))
        
        self.bins = bins
        self.ensemble_name = ensemble_name

        # helpful to store for plotting or other analysis
        histograms = []
        entropies = []
        for obs in self.observations:
            hist, nbins = np.histogram(obs, bins=self.bins)
            histograms.append(hist)
            entropies.append(shannon_entropy(hist, True))
        
        self.histograms = np.array(histograms)
        self.entropies = np.array(entropies)


    """
    Calculations & metrics
    
    """

    @property
    def ensemble(self):
        return np.mean(self.entropies)

    @property
    def geometric_ensemble_mean(self):
        return self.entropies.prod()**(1.0/len(self.entropies))

    @property
    def ergodic(self):
        try:
            return self._ergodic
        except AttributeError:
            hist, nbins = np.histogram(np.concatenate(self.observations), bins=self.bins)
            self._ergodic = shannon_entropy(hist, True)
            return self._ergodic

    @property
    def complexity(self):
        return complexity(self.ensemble, self.ergodic)



    """
    Plot
    
    """

    def plot(self):
        # in function import so doesn't require
        # pandas or seaborn to use above
        from .plots import dual
        en = 'ensemble'
        if self.ensemble_name is not None:
            en = self.ensemble_name
        dual(self.observations, self.bins, self.labels, variable=en)

    def ridge(self):
        from .plots import ridge
        ridge(self.observations, self.bins)
        
        
    def stats(self):
        msg = ""
        if self.ensemble_name is not None:
            msg += "%s\n" % self.ensemble_name
        msg += "%.1f%% ergodic complexity\n" % (self.complexity*100)
        msg += "%.3f (%.3f) average ensemble (ergodic)\n" % (self.ensemble, self.ergodic)
        msg += "From %s ensembles\n" % len(self.observations) 
        msg += "With bins %s from %s to %s.\n" % (len(self.bins), self.bins[0], self.bins[-1])
        print(msg)
