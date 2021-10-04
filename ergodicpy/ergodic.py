import numpy as np

from .entropy import measures, LEGEND, observation_weights
from .bins import binobs, ergodic_obs


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
    
    :bins: _None_. the bins to be used for the data
    if :bins: is None, then it assumes system is continous and automatically finds optimum

    :labels: the names of the ensembles for plotting
    :ensemble_name: the name of the ensemble to be used plots
    :dist_name: the name of the distribution variable
    :units: 'bits' or 'nats' of entropy
    :lazy: _False_ will calculate measures on creation if False, otherwise need to call `analyse()`

    properties
    :measures: all the four measures below, which are also assigned as properties
    :ensemble: the average ensemble entropy
    :ergodic: the entropy of the ergodic distribution
    :divergence: the divergence metric
    :complexity: the complexity metric

    :histograms: the histograms of each ensemble
    :entropies: the entropy for each histogram

    functions
    :stats: a dict of the key statistics
    :plot: plots the ensemble & ergodic histograms
    :ridge: plots a ridge plot of the ensemble histograms
    :scatter: plots a scatter graph with data approximated into bins
    :stats: prints all the stats in an easy to read format
    """
    def __init__(self, observations, bins=None, weights=None,
                labels=None, ensemble_name='ensemble', dist_name='value',
                    units=None, tau_boost=None, lazy=False):

        # handle observations
        self.observations = observations
        self.ergodic_observations = ergodic_obs(self.observations)

        # if distribution is continuous and need to define bins here
        if bins is None:
            bins = binobs(observations)
        self.bins = bins

        # default weights of N_k / N
        self.weights = None

        # 'bits' or 'nats' of shannon entropy
        self.units = units
        self.tau_boost = tau_boost

        # naming for plots
        self.labels = labels
        self.ensemble_name = ensemble_name
        self.dist_name = dist_name
        

        # do analysis
        if not lazy:
            self.analyse()


    """
    Essential calculations & metrics
    
    """

    def analyse(self):
        """
        Does all the analysis.
        Easiest way to compare different input values e.g. continuous or not,
        number of bins etc, is to reset on the object then recall analyse().
        """
        # calc histograms as pmfs
        histograms = []
        for obs in self.observations:
            # ignore erroroneous ensembles with no observations
            if len(obs) > 0:
                hist, nbins = np.histogram(obs, bins=self.bins)
                histograms.append(hist)
        self.histograms = np.array(histograms)

        # get measures
        ms = measures(self.histograms, weights=self.weights,
            units=self.units, boost=self.tau_boost, with_meta=True)

        for k, v in ms.items():
            setattr(self, k, v)

        del ms['entropies']
        del ms['weights']
        self.measures = ms
        
    
    """
    Metrics
    
    """
    @property
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

    @property
    def ensemble_count(self):
        """ Total number of ensembles """
        return len(self.observations)


    """
    Comparison metrics

    """
    def chi2(self):
        from scipy.stats import chi2_contingency
        # returns (chi2, p, dof, expected)
        return chi2_contingency(self.histograms)

    """
    Plots & displays
    
    """

    def ensemble_melt(self):
        """ Dataframe of ensemble data prepared for plots """
        import pandas as pd
        return pd.melt(pd.DataFrame(self.observations, index=self.labels).T,
            var_name=self.ensemble_name, value_name=self.dist_name)

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
        dual(self.ensemble_melt(), self.ergodic_melt(), self.bins, self.labels,
            tidy_variable=self.ensemble_name, tidy_value=self.dist_name, title=title)

    def ridge(self):
        from .plots import ridge
        ridge(self.ensemble_melt(), self.bins, self.labels,
            tidy_variable=self.ensemble_name, tidy_value=self.dist_name)
        
    def scatter(self):
        from .plots import scatter
        scatter(self.ensemble_melt(), self.bins,
            tidy_variable=self.ensemble_name, tidy_value=self.dist_name)
