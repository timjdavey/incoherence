import seaborn as sns
from .series import RIGHT_GRAPH


class ErgodicScan(dict):
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
    def __init__(self, x, y, trend=0.1):
        self.x = x
        self.y = y
        self.trend = trend

        for i, key in enumerate(x):
            self[key] = y[i]

    @property
    def steps(self):
        """ For calculating trend"""
        return len(self.y[0].complexities)

    @property
    def trend_from(self):
        return int(self.steps*(1-self.trend))

    def plot(self, lines=RIGHT_GRAPH[1:]):
        """
        :lines: which measures & colors you want to plot
        :returns: the figure so you can alter title, xlim etc
        """
        for name, color, alt_color in lines:
            maximums = [e.measures[name].max() for e in self.y]
            fig = sns.lineplot(x=self.x, y=maximums, label='%s (max)' % name, color=alt_color)
            
            trends = [e.measures[name][self.trend_from:].mean() for e in self.y]
            sns.lineplot(x=self.x, y=trends, label='%s (trend)' % name, color=color)
        
        # move legend outside of box, as it's huge
        fig.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        return fig