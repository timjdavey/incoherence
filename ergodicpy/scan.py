import seaborn as sns
from .entropy import LEGEND


class ErgodicScan:
    """
    Plots the maximum & trend complexities for a series of variables.
    This is a kind of a 3D plot (but more helpful as contains multiple surfaces).
    ErgodicSeries can do a scan for multiple variables, but if the model
    depends on the system evolving, typically you'll need to look at
    the maximum complexity and trend over time.

    Trend here is the mean of the last 10% of observations.
    This is a class so can store variables easier.
    
    :x: the x axis variables which were altered.
    :y: a list of ErgodicSeries.
    :trend: "0.1" percentage of final observations to take as trend.
    
    .plot() :returns: the figure so you can adjust limits, title etc.

    :returns: is a dict class, so can access individual series through keys
    """
    legend_keys = ('divergence', 'complexity')

    def __init__(self, x, y, trend=0.1):
        self.x = x
        self.y = y
        self.trend = trend
        self.lines = []

        self.analyse()

    def analyse(self):
        for key in self.legend_keys:
            self.lines['%s max' % key] = [e.measures[key].max() for e in self.y]
            self.lines['%s trend' % key] = [e.measures[key][self.trend_from:].mean() for e in self.y]

    @property
    def steps(self):
        """ For calculating trend"""
        return len(self.y[0].x)

    @property
    def trend_from(self):
        """ The position of the [i:] slice """
        return int(self.steps*(1-self.trend))
    

    def plot(self, legend=LEGEND):
        """
        :lines: which measures & colors you want to plot
        :returns: the figure so you can alter title, xlim etc
        """
        for key in legend_keys:
            for version, color in ("max", "alt"), ("trend", "color"):
                fig = sns.lineplot(x=self.x, y=self.lines['%s %s' % (key, version)],
                    label='%s (%s)' % (legend[key]['verbose'], version), color=color)
        
        # move legend outside of box, as it's huge
        fig.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        return fig