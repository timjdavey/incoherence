import numpy as np
from functools import cached_property

from .bins import binr
from .ergodic import ErgodicEnsemble
from .entropy import LEGEND


class ErgodicSeries:
    """
    Simple class to handle the ergodic analysis over a series of values.
    """
    def __init__(self, x=None, y=None, observations=None, x_label='x', title=None, bins=None, units='nats', models=None):
        self.x = x
        self.x_label = x_label

        # either or, checked later
        self.y = y
        self.observations = observations

        self.title = title
        self.units = units
        self.bins = bins
        self.models = models
        self.map = {}

        self.analyse()

    def analyse(self):
        """
        Creates ErgodicEnsembles for each series (typically time)
        and stores the analysis for plotting
        """
        
        # type check
        if self.y is None and self.observations is None:
            raise InputError("Please supply a list of ErgodicEnsembles through `y` or raw `observations`")
        
        # create ergodics if needed
        elif self.observations is not None:
            # ideally bins are consistent across all of series
            if self.bins is None:
                self.bins = binr(series=self.observations)
            self.y = [ErgodicEnsemble(obs, self.bins) for obs in self.observations]

        # type check y otherwise
        elif not isinstance(self.y, (list, np.ndarray)) or not isinstance(self.y[0], (ErgodicEnsemble)):
            raise TypeError("`y` should be a `list` of `ErgodicEnsemble`")

        # create x if needed, defaulted to just a count
        if self.x is None:
            self.x = range(len(self.y))


        # analyse and store across all steps
        entropies, measures, raw = [], [], []
        for ee in self.y:
            # analyse
            ms = ee.measures
            measures.append(list(ms.values()))
            raw.append(ms)
            entropies.append(ee.entropies)
        
        # store
        self.entropies = np.array(entropies)
        self.raw = raw
        
        # store measures as columns of {[],[]}
        self.measures = {}
        stacked = np.stack(np.array(measures), axis=1)
        for i, k in enumerate(self.y[0].measures.keys()):
            self.measures[k] = stacked[i]

        # dict access to each ErgodicEnsemble
        for i, x in enumerate(self.x):
            self.map[x] = self.y[i]

    """
    Stats
    """
    def _sf(self, arr, percent):
        """ slice_from The position of the [i:] slice """
        return arr[int(len(self.x)*(1-percent)):]
    
    def max(self, p=1.0):
        return dict([(k, self._sf(v, p).max()) for k, v in self.measures.items()])

    def trend(self, p=0.1):
        return dict([(k, self._sf(v, p).mean()) for k, v in self.measures.items()])

    """
    Data Anaylsis
    """
    def dataframe(self):
        """ Simply ceates a dataframe, but imports pandas so whole module doesn't rely on it """
        import pandas as pd
        df = pd.DataFrame(data=self.raw)
        df[self.x_label] = self.x
        return df

    def to_dict(self, keys=None):
        if keys is None:
            keys = ('x', 'x_label', 'title', 'units', 'bins') # no y
        data = []
        for key in keys:
            attr = getattr(self, key)
            if isinstance(attr, np.ndarray):
                attr = attr.tolist()
            data.append((key, attr))
        return dict(data)

    def bin_stats(self):
        """ Stats about the bins which were used """
        return "%s bins from %s to %s" % (len(self.bins)-1, self.bins[0], self.bins[-1])

    @property
    def peaks(self):
        return np.where(self.measures['complexity'] == np.amax(self.measures['complexity']))[0]

    def results(self, steps=False):
        """ Outputs most of the really interesting analysis """
        self.plot()
        print("Maximums:", self.max())
        print("Trending to:", self.trend())
        
        if steps:
            # first results
            self.step_plot(0, "first")
            # first maximum plot
            self.step_plot(self.peaks[0], "max complexity")
            # last results
            self.step_plot(-1, "last")     

    """
    Plotting
    """
    def _lineplot(self, key, ax, ylabel=None, ymaxmin=None):
        """ Internal function to properly format the lines """
        import seaborn as sns
        y = np.array(self.measures[key])
        g = sns.lineplot(x=self.x, y=y, ax=ax,
                label=LEGEND[key][0], color=LEGEND[key][1])
        g.set_xlabel(self.x_label)
        g.set_ylabel(ylabel)

        # give reasonable defaults to graphs for sense of scale
        if ymaxmin:
            g.set(ylim=(min(-ymaxmin*0.05,y.min()), max(ymaxmin,y.max())))
        return g

    def plot(self, ymax=0.5):
        """ Plots the evolution of the entropies & complexities over time """
        import seaborn as sns
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(15,5))
        # ensembles
        ensemble_entropies = np.stack(self.entropies, axis=1)

        # rotate through mako colours, as not using pandas dataframe doing manually
        ncolors = ensemble_entropies.shape[0]
        palette = sns.light_palette("skyblue", ncolors, reverse=True)
        for i, e in enumerate(ensemble_entropies):
            sns.lineplot(x=self.x, y=e, ax=axes[0], color=palette[i])

        # ergodic & ensemble
        g = self._lineplot('ensemble', axes[0], 'Entropy (%s)' % self.units, 1.0)
        g = self._lineplot('ergodic', axes[0], 'Entropy (%s)' % self.units, 1.0)
        g.set_title("Entropies (incl semi-transparent blue individual ensembles)")
        

        # second plot
        ax2 = axes[1].twinx()
        h = self._lineplot('complexity', axes[1], 'Complexity (%s)' % self.units, ymax)
        j = self._lineplot('divergence', ax2, 'Divergence (%s)' % self.units, ymax)
        j.set_title(self.title if self.title else "Ergodic complexity and divergence")

        # combine legends
        h1, l1 = axes[1].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        axes[1].legend(h1+h2, l1+l2, loc=0)
        ax2.get_legend().remove()
        
        return fig

    def step_plot(self, index, comment=""):
        """ Plot a specific ErgodicEnsemble step """
        title = "%s=%s, step=%s (%s)" % (self.x_label, self.x[index], index, comment)
        self.y[index].plot(title)
