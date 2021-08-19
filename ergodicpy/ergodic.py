import numpy as np
from functools import cached_property

from .entropy import measures
from .bins import list_observations, binr



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
    
    :bins: _None_. the bins to be used for the data, suggest using ep.binr
    :ensemble_name: the name of the ensemble to be used plots
    :dist_name: the name of the distribution variable

    properties
    :ensemble: the average ensemble entropy
    :ergodic: the entropy of the ergodic distribution
    :divergence: the divergence metric
    :complexity: divergence divided by ergodic

    functions
    :plot: plots the ensemble & ergodic histograms
    :ridge: plots a ridge plot of the ensemble histograms
    :stats: prints all the stats in an easy to read format
    """
    def __init__(self, observations, bins=None, 
            ensemble_name='ensemble', dist_name='value', units=None):

        # observations by dict or list
        self.raw = observations
        self.observations, self.labels = list_observations(observations)

        # auto create bins
        if bins is None:
            self.bins = binr(observations=observations)
        else:
            self.bins = bins

        # 'bits' or 'nats' of shannon entropy
        self.units = units

        # naming for plots
        self.ensemble_name = ensemble_name
        self.dist_name = dist_name


    """
    Essential calculations & metrics
    
    """
    @cached_property
    def histograms(self):
        """ List of histograms of each ensemble """
        histograms = []
        for obs in self.observations:
            # ignore erroroneous ensembles with no observations
            if len(obs) > 0:
                hist, nbins = np.histogram(obs, bins=self.bins)
                histograms.append(hist)
        return np.array(histograms)

    @cached_property
    def measures(self):
        ms = measures(self.histograms, True, self.units, with_entropies=True)
        self._entropies = ms['entropies of ensembles']
        del ms['entropies of ensembles']
        return ms

    @property
    def complexity(self):
        return self.measures['ergodic complexity (2)']

    @property
    def entropies(self):
        self.measures # is cached
        return self._entropies


    """
    Key metrics
    
    """
    
    @cached_property
    def obs_counts(self):
        """ The min, mean and max observations across ensembles """
        # store data about observation counts across ensembles
        # can't use shape, as can be different lengths
        obs_counts = [len(o) for o in self.observations]
        return {
            'minimum':np.amin(obs_counts),
            'mean': np.mean(obs_counts),
            'max': np.amax(obs_counts),
        }

    @cached_property
    def ensemble_count(self):
        """ Total number of ensembles """
        return len(self.observations)


    def stats(self, display=False):
        measures = self.measures
        measures['ensembles count'] = self.ensemble_count
        measures['entropies'] = self.entropies
        measures['bins count'] = len(self.bins)-1
        measures['bins range'] = (self.bins[0], self.bins[-1])
        measures['observations'] = self.obs_counts
        if display:
            for k,v in measures.items():
                print(v,k)
        else:
            return measures

    """
    Plots & displays
    
    """

    @cached_property
    def ergodic_observations(self):
        """ All the observations as one """
        return np.concatenate(self.observations)

    @cached_property
    def ensemble_melt(self):
        """ Dataframe of ensemble data prepared for plots """
        import pandas as pd
        return pd.melt(pd.DataFrame(self.observations, index=self.labels).T,
            var_name=self.ensemble_name, value_name=self.dist_name)

    @cached_property
    def ergodic_melt(self):
        """ Dataframe of ergodic data prepared for plots """
        import pandas as pd
        return pd.DataFrame({
            self.dist_name:self.ergodic_observations,
            self.ensemble_name:'h',})

    def plot(self, title=None):
        # import in function import so doesn't require
        # pandas or seaborn to use above
        from .plots import dual
        dual(self.ensemble_melt, self.ergodic_melt, self.bins, self.labels,
            tidy_variable=self.ensemble_name, tidy_value=self.dist_name, title=title)

    def ridge(self):
        from .plots import ridge
        ridge(self.ensemble_melt, self.bins, self.labels,
            tidy_variable=self.ensemble_name, tidy_value=self.dist_name)
        
    def scatter(self):
        from .plots import scatter
        scatter(self.ensemble_melt, self.bins,
            tidy_variable=self.ensemble_name, tidy_value=self.dist_name)
