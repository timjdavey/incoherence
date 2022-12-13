import numpy as np
from functools import cached_property

from .continuous import Continuous
from .entropy import point_pmf
from .divergences import kl_divergences
from .stats import measures, LEGEND
from .bins import binint, binspace, binobs, pooled_obs


class Discrete(Continuous):
    """
    A base model to help calculate the various measures.

    Contains some simple performance boosts, but also stores
    some helpful data to make visualisation simpler.
    
    intialization
    :observations: either takes a list observations grouped by ensemble
    :bins: _None_. the bins to be used for the data
    :weights: _None_, list, for the ensembles
    :labels: the names of the ensembles for plotting
    :ensemble_name: the name of the ensemble to be used plots
    :dist_name: the name of the distribution variable
    :base: _None_ units default is natural e units of entropy
    :lazy: _False_ will calculate measures on creation if False, otherwise need to call `analyse()`
    """
    def __init__(self, observations, bins, weights=None, metrics=None,
                labels=None, base=None, lazy=False):

        self.discrete = True
        self.histograms = None
        self.observations = observations
        self.bins = bins
        self.weights = weights
        self.base = base
        self.labels = labels
        self.metrics = None

        # do analysis
        if not lazy:
            self.analyse()

    def _get_measures(self):
        """
        Gets the standard measures, now using histograms.
        """
        if self.histograms is None:
            histograms = []
            for obs in self.observations:
                # ignore erroroneous ensembles with no observations
                if len(obs) > 0:
                    hist, nbins = np.histogram(obs, bins=self.bins)
                    histograms.append(hist)
            self.histograms = np.array(histograms)

        return measures(self.histograms, weights=self.weights,
                base=self.base, discrete=True, metrics=self.metrics)
        
    """
    Helper Metrics
    
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
        return len(self.histograms)

    @cached_property
    def pooled_observations(self):
        return pooled_obs(self.observations)

    @cached_property
    def obs_min(self):
        return self.pooled_observations.min()

    @cached_property
    def obs_max(self):
        return self.pooled_observations.max()

    @property
    def states(self):
        if self.bins is not None:
            return len(self.bins)-1
        elif self.histograms is not None:
            return len(self.histograms[0])
        else:
            return None
    
    """
    Comparison metrics

    """
    def chi2(self, ignore=False):
        from scipy.stats import chi2_contingency
        # returns (chi2, p, dof, expected)
        try:
            return chi2_contingency(self.histograms)
        except ValueError:
            return None, None, None, None

    def ergodic_pmf(self):
        """
        The ergodic ensemble (pmf as normalised).
        With standard weights.
        """
        return point_pmf(self.histograms, weights=self.weights)

    def obs_to_hist(self, observations):
        """
        Returns a histogram, for a given set of `observations`
        using the current set of bins.
        """
        return np.histogram(observations, self.bins)



# handling legacy
Ensembles = Discrete


def histogram_to_observations(histogram, states=None):
    """
    Converts a histogram to a series of raw observations.
    Mainly for use with plots.
    """
    if states is None:
        states = range(len(histogram))

    # it is a pmf with some error
    boost = 1
    if np.sum(histogram) < 2:
        # given is really only for plots doesn't need to be crazy accurate
        boost = 10000
    return np.concatenate([np.ones(int(volume*boost))*states[state]
        for state, volume in enumerate(histogram)])








