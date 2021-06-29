import numpy as np
from functools import cached_property

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
    :dist_name: the name of the distribution variable

    properties
    :ensemble: the average ensemble entropy
    :ergodic: the entropy of the ergodic distribution
    :complexity: the ergodic complexity

    functions
    :plot: plots the ensemble & ergodic histograms
    :ridge: plots a ridge plot of the ensemble histograms
    :stats: prints all the stats in an easy to read format
    """
    def __init__(self, observations, bins, ensemble_name='ensemble', dist_name='value'):

        # naming for plots
        self.ensemble_name = ensemble_name
        self.dist_name = dist_name

        # observations by dict or list
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
        
        # all of the observations
        self.ergodic_observations = np.concatenate(self.observations)

        # error check bins must be constructed correctly
        emin = self.ergodic_observations.min()
        emax = self.ergodic_observations.max()
        if len(bins) < 2:
            raise ValueError("%s bins is too small" % bins)
        elif bins[0] > emin:
            raise ValueError("%s lower bin value less than ergodic min" % emin)
        elif bins[-1] < emax:
            raise ValueError("%s higher bin value less than ergodic max" % emax)
        else:
            self.bins = bins



    """
    Data processing
    
    """
    @cached_property
    def entropies(self):
        """ Array of entropies for each ensemble """
        entropies = []
        for obs in self.observations:
            hist, nbins = np.histogram(obs, bins=self.bins)
            entropies.append(shannon_entropy(hist, True))
        return np.array(entropies)

    @cached_property
    def ensemble_melt(self):
        """ Dataframe of ensemble data prepared """
        import pandas as pd
        return pd.melt(pd.DataFrame(self.observations, index=self.labels).T,
            var_name=self.ensemble_name, value_name=self.dist_name)

    @cached_property
    def ergodic_melt(self):
        import pandas as pd
        return pd.DataFrame({
            self.dist_name:self.ergodic_observations,
            self.ensemble_name:'h',})
    
    """
    Calculations & metrics
    
    """
    @cached_property
    def ensemble_count(self):
        return len(self.entropies)

    @cached_property
    def ensemble(self):
        """ The average (mean) ensemble entropy """
        return np.mean(self.entropies)

    @cached_property
    def geometric_ensemble_mean(self):
        """ The geometric (mean) ensemble entropy """
        return self.entropies.prod()**(1.0/self.ensemble_count)

    @cached_property
    def ergodic(self):
        """ The entropy of the ergodic distribution """
        hist, nbins = np.histogram(self.ergodic_observations, bins=self.bins)
        return shannon_entropy(hist, True)

    @cached_property
    def complexity(self):
        """ The ergodic complexity """
        return complexity(self.ensemble, self.ergodic)



    """
    Plots & displays
    
    """

    def plot(self):
        # in function import so doesn't require
        # pandas or seaborn to use above
        from .plots import dual
        dual(self.ensemble_melt, self.ergodic_melt, self.bins, self.labels,
            tidy_variable=self.ensemble_name, tidy_value=self.dist_name)

    def ridge(self):
        from .plots import ridge
        ridge(self.ensemble_melt, self.bins, self.labels,
            tidy_variable=self.ensemble_name, tidy_value=self.dist_name)
        
    def scatter(self):
        from .plots import scatter
        scatter(self.ensemble_melt, self.bins,
            tidy_variable=self.ensemble_name, tidy_value=self.dist_name)
    
    def stats(self):
        msg = ""
        if self.ensemble_name is not None:
            msg += "%s\n" % self.ensemble_name
        msg += "%.1f%% ergodic complexity\n" % (self.complexity*100)
        msg += "%.3f (%.3f) average ensemble (ergodic)\n" % (self.ensemble, self.ergodic)
        msg += "From %s ensembles\n" % self.ensemble_count
        msg += "With bins %s from %s to %s.\n" % (len(self.bins), self.bins[0], self.bins[-1])
        print(msg)
