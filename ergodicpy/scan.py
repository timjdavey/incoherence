import numpy as np

from .entropy import LEGEND
from .series import ErgodicSeries



class ErgodicScan:
    """
    Plots the maximum & trend complexities for a series of variables.
    This is a kind of a 3D plot (but more helpful as contains multiple surfaces).
    ErgodicSeries can do a scan for multiple variables, but if the model
    depends on the system evolving, typically you'll need to look at
    the maximum complexity and trend over time.

    Trend here is the mean of the last 10% of observations.
    This is a class so can store variables easier.
    
    Inputs
    :x: the x axis variables which were altered.
    :y: a list of ErgodicSeries.
    :x_label: for the plot access.
    :title: for the plot.
    :trend: "0.1" percentage of final observations to take as trend.
    :max_min: Same as trend, what final observations should you take to find max.
    
    .plot() :returns: the figure so you can adjust limits, title etc.

    :returns: is a dict class, so can access individual series through keys
    """

    def __init__(self, x=None, y=None, x_label='x', title=None, trend=0.1, max_trend=1.0):

        # check passed values & loading went correctly
        if not isinstance(y, (list, np.ndarray)) or not isinstance(y[0], (ErgodicSeries)):
            raise TypeError("`y` should be a `list` of `ErgodicSeries`")
        else:
            self.y = np.array(y)

        # default x to just a range
        self.x = range(len(self.y)) if x is None else np.array(x)
        self.x_label = x_label
        self.title = title
        self.trend = trend
        self.max_trend = max_trend

        self.measures = {}
        self.map = {}

        # store the key measures
        for key in ('ensemble', 'ergodic', 'divergence', 'complexity'):
            self.measures['%s max' % key] = [e.max(max_trend)[key] for e in self.y]
            self.measures['%s trend' % key] = [e.trend(trend)[key] for e in self.y]

        # dict access to each ErgodicSeries
        for i, x in enumerate(self.x):
            self.map[x] = self.y[i]
    
    """
    Data analysis
    """
    def dataframe(self):
        """ Returns the analysis as a dataframe """
        import pandas as pd
        return pd.DataFrame(data=self.measures, index=self.x)

    def to_dict(self, keys=None):
        if keys is None:
            keys = ('x', 'x_label', 'title', 'trend', 'max_trend', 'measures') # leaves out y
        return dict([(k, getattr(self, k)) for k in keys])
    
    """
    Plotting
    """
    def _lineplot(self, key, ax, ylabel=None, ymaxmin=None):
        """ Internal function to properly format the lines """
        import seaborn as sns
        # trend
        g = sns.lineplot(x=self.x, y=self.measures["%s trend" % key], ax=ax,
                label=LEGEND[key][0], color=LEGEND[key][1])
        # max
        y = np.array(self.measures["%s max" % key])
        g = sns.lineplot(x=self.x, y=y, ax=ax,
                label='%s max' % key, color=LEGEND[key][1], alpha=0.5, linestyle="dotted")
        
        # labels
        g.set_xlabel(self.x_label)
        g.set_ylabel(ylabel)

        # give reasonable defaults to graphs for sense of scale
        if ymaxmin:
            g.set(ylim=(min(-ymaxmin*0.05,y.min()), max(ymaxmin,y.max())))
        return g

    def plot(self, ymax=0.2):
        """
        :returns: the figure so you can alter title, xlim etc
        """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(15,5))

        # ergodic & ensemble
        g = self._lineplot('ensemble', axes[0], 'Entropy', 1.0)
        g = self._lineplot('ergodic', axes[0], 'Entropy', 1.0)
        g.set_title("Entropies")
        

        # second plot
        ax2 = axes[1].twinx()
        h = self._lineplot('complexity', axes[1], 'Complexity', ymax)
        j = self._lineplot('divergence', ax2, 'Divergence', ymax)
        j.set_title(self.title if self.title else "Ergodic complexity and divergence")

        # combine legends
        h1, l1 = axes[1].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        axes[1].legend(h1+h2, l1+l2, loc=0)
        ax2.get_legend().remove()
        return fig

