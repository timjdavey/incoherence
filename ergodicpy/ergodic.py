import numpy as np

from .entropy import measures
from .bins import bino



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
    def __init__(self, observations, bins=None, labels=None, weights=None,
            ensemble_name='ensemble', dist_name='value', units=None, lazy=False):

        # auto create bins
        self.observations = observations
        self.ergodic_observations = np.concatenate(observations)
        
        # default weights of N_k / N
        if weights is None:
            total = len(self.ergodic_observations)
            weights = [len(e)/total for e in observations]
        self.weights = weights

        # default bins set in binr
        self.bins = bins # these are set properly later

        # 'bits' or 'nats' of shannon entropy
        self.units = units

        # naming for plots
        self.labels = labels
        self.ensemble_name = ensemble_name
        self.dist_name = dist_name

        # do analysis
        if not lazy:
            if bins is None:
                self.stablize()
            else:
                self.analyse()


    """
    Essential calculations & metrics
    
    """
    @property
    def ergodic_histogram(self):
        return np.sum(self.histograms, axis=0)

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
            units=self.units, with_entropies=True)

        for k, v in ms.items():
            setattr(self, k, v)
        del ms['entropies']
        self.measures = ms

    def _bin_search(self, xs):
        """
        For a given bin count range `xs`
        Returns the `optimium_count` of bins and a 2D array of `bin` to `complexity` value
        """
        indx = []

        for x in xs:
            self.bins = bino(self.observations, x)
            self.analyse()
            indx.append([x,self.complexity])

        indx = np.array(indx)
        ys = np.array(indx[:,1])
        optimum_index = np.where(ys==ys.min())[0][0]
        return optimum_index, indx

    def _bin_optimize(self, minimum, maximum, spread, depth, iteration=0):
        """
        A faster, recursive approach to finding optimum bins than searching through entire range.
        Doesn't have to be perfect, since complexity levels off for such a large range.
        """
        xs = np.unique([int(i) for i in np.linspace(minimum,maximum,spread)])
        optimum_index, indx = self._bin_search(xs)

        if iteration < depth:
            try:
                return self._bin_optimize(indx[optimum_index-1][0], indx[optimum_index+1][0], spread, depth, iteration+1)
            except IndexError:
                return optimum_index, indx
        else:
            return optimum_index, indx

    def stablize(self, minimum=None, maximum=None, update=True, optimized=True, plot=False, spread=20, depth=20):
        """
        If dealing with a continuous distribution,
        finds the optimum bin count.

        :minimum: 3, minimum bin range to explore from
        :maximum: observations/20 of the range to explore
        :reset: 
        """
        legacy = self.bins

        # set defaults
        # need minmum 3 bins (small & odd)
        if minimum is None: minimum = 3
        # need at least 7 per bin
        if maximum is None: maximum = max(4,int(self.obs_counts['mean']/7))

        # explore entire bin range for scan
        if plot or not optimized:
            optimum_index, indx = self._bin_search(range(minimum,maximum))
        else:
            optimum_index, indx = self._bin_optimize(minimum, maximum, spread, depth)

        # update the bins or reset to original
        if update:
            self.bins = bino(self.observations, indx[optimum_index][0])
        else:
            self.bins = legacy
        self.analyse()

        # plot the results if needed
        if plot:
            import seaborn as sns
            sns.lineplot(x=indx[:,0], y=indx[:,1])

        return optimum_index, indx


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
    Comparison metrics
    """
    def chi2(self, ignore=True):
        from scipy.stats import chi2_contingency
        try:
            # returns (chi2, p, dof, expected)
            return chi2_contingency(self.histograms)
        # throws exception when there's a zero in histogram
        # this is surprisingly often
        except ValueError:
            # rather than halting the programme
            # return None and handle seperately
            if ignore:
                return (None, None, None, None)
            else:
                raise ValueError(e)


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
