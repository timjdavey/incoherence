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
                labels=None, ensemble_name='ensemble', dist_name='value',
                base=None, lazy=False):

        self.discrete = True
        self.histograms = None
        self.observations = observations
        self.bins = bins
        self.weights = weights
        self.base = base
        self.labels = labels
        self.ensemble_name = ensemble_name
        self.dist_name = dist_name
        self.metrics = None

        # do analysis
        if not lazy:
            if bins is None:
                self.stabilize()
            else:
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


    """
    Finding optimum bins for continuous entropy distributions
    
    """

    def _bin_search(self, xs):
        """
        For a given bin count range `xs`
        Returns the `optimium_count` of bins and a 2D array of `bin` to `incoherence` value
        """
        indx = []

        for x in xs:
            self.update_bins(x)
            indx.append([x, self.incoherence])

        indx = np.array(indx)
        ys = np.array(indx[:,1])
        optimum_index = np.where(ys==ys.min())[0][0]
        return optimum_index, indx

    def _bin_optimize(self, minimum, maximum, spread, depth, iteration=0, base_threshold=0.0001):
        """
        A faster, recursive approach to finding optimum bins than searching through entire range.
        Doesn't have to be perfect, since incoherence levels off for such a large range.
        """
        xs = binint(minimum, maximum, spread)
        optimum_index, indx = self._bin_search(xs)
        optimum_bin = indx[optimum_index][0]
        upper_index = len(indx)-1
        
        # return if reached max depth
            # or incoherence is basically zero
                # or the difference is basically zero
                    # you're looking at nearest neighbours
        if iteration >= depth \
            or indx[optimum_index][1] < base_threshold \
                or (optimum_index > 0 and indx[optimum_index-1][0] == optimum_bin-1) \
                    or (optimum_index < upper_index and indx[optimum_index+1][0] == optimum_bin+1):
            
            return optimum_index, indx
        else:
            # use the lowest & highest available
            lower = 0 if optimum_index == 0 else optimum_index-1
            upper = optimum_index if optimum_index == upper_index else optimum_index+1
            return self._bin_optimize(indx[lower][0], indx[upper][0],
                spread, depth, iteration+1, base_threshold=base_threshold)
            

    def stabilize(self, minimum=None, maximum=None, optimized=True, plot=False, spread=4, depth=10, ax=None):
        """
        If dealing with a continuous distribution,
        finds the optimum bin count.
        :minimum: 3, minimum bin range to explore from
        :maximum: observations/5 of the range to explore
        :update: _True_ update to the optimimum bins or return to current
        :optimized: _True_ use a faster search version
        :plot: _False_ plot a figure of bin number against incoherence at that bin
        :spread: 4, the number of bins to try during optimised
        :depth: 10, the max number of depth searches to be used during optimised search
        :cheat: _True_, cheat mode which just uses obs/5 bins
        """
        # set defaults
        # need minmum 3 bins (lowest odd)
        if minimum is None: minimum = 3
        # need at least 5 per bin or 6 bins
        if maximum is None: maximum = max(6,int(self.obs_counts['mean']/5))

        # explore entire bin range for scan
        if plot or not optimized:
            optimum_index, indx = self._bin_search(binint(minimum,maximum))
        else:
            optimum_index, indx = self._bin_optimize(minimum, maximum, spread, depth)

        # update the bins or reset to original
        self.update_bins(indx[optimum_index][0])

        # plot the results if needed
        if plot:
            import seaborn as sns
            g = sns.lineplot(x=indx[:,0], y=indx[:,1], color='blue', ax=ax)
            g.set(ylim=(0, None))

        # get optimum bins by indx[optimum_index][0] or len(self.bins)-1
        return optimum_index, indx

    def update_bins(self, bin_count, obs_min=None, obs_max=None):
        """
        Resets the bins to a specific count
        """
        if obs_min is None: obs_min = self.obs_min
        if obs_max is None: obs_max = self.obs_max
        self.bins = binspace(obs_min, obs_max, bin_count)
        self.analyse()
        return self

    """
    Self organisation

    """
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

    def bayes_posterior(self, histogram):
        """
        Given a `histogram`,
        Returns the full bayesian posterior.
        Using the ensembles as likelihood functions.
        """
        if len(histogram) != self.states:
            raise ValueError("histogram does not have correct states %s != %s" % (len(histogram), self.states))

        prior = np.array(self.weights)
        data = np.array(histogram)

        likelihood = np.array([np.product(l**data) for l in self.histograms])
        posterior = likelihood*prior
        return posterior/posterior.sum()

    def bayes_pmf(self, histogram, references=None, with_posterior=False):
        """
        Given a `histogram` of observational data,
        Returns the predicted bayesian pmf.
        Using the ensemble histograms as the references
        """
        posterior = self.bayes_posterior(histogram)
        if references is None: references = self.histograms
        pmf = point_pmf(references, weights=posterior)

        if with_posterior:
            return pmf, posterior
        else:
            return pmf
    

    """
    Plots & displays
    
    """

    def ensemble_melt(self):
        """ Dataframe of ensemble data prepared for plots """
        import pandas as pd
        return pd.melt(pd.DataFrame(self.observations, index=self.labels).T,
            var_name=self.ensemble_name, value_name=self.dist_name)

    def pooled_melt(self):
        """ Dataframe of pooled data prepared for plots """
        import pandas as pd
        return pd.DataFrame({
            self.dist_name:self.pooled_observations,
            self.ensemble_name:'h',})

    def plot(self, title=None):
        # import in function import so doesn't require
        # pandas or seaborn to use above
        from .plots import dual
        return dual(self.ensemble_melt(), self.pooled_melt(), self.bins, self.labels,
            tidy_variable=self.ensemble_name, tidy_value=self.dist_name, title=title)

    def ridge(self):
        from .plots import ridge
        return ridge(self.ensemble_melt(), self.bins, self.labels,
            tidy_variable=self.ensemble_name, tidy_value=self.dist_name)
        
    def scatter(self):
        from .plots import scatter
        return scatter(self.ensemble_melt(), self.bins,
            tidy_variable=self.ensemble_name, tidy_value=self.dist_name)


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








