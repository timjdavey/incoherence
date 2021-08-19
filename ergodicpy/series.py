import numpy as np

from .bins import binr
from .ergodic import ErgodicEnsemble


class ErgodicSeries:
    """
    Simple class to handle the ergodic analysis over a series of values.
    """
    def __init__(self, x, y, x_label='x', title=None, bins=None, units=None):
        self.x = x
        self.x_label = x_label
        self.title = title
        self.y = y
        self.units = units
        self.bins = bins

        self.analyse()

    def analyse(self):
        """
        Creates ErgodicEnsembles for each series (typically time)
        and stores the analysis for plotting
        """
        # need to make sure bins are consistent across all of series
        if self.bins is None:
            self.bins = binr(series=self.y)

        # analyse and store across all steps
        ees, entropies, measures, raw = [], [], [], []
        for i, timestep_obs in enumerate(self.y):
            # attach
            ee = ErgodicEnsemble(timestep_obs, self.bins)
            ees.append(ee)
            entropies.append(ee.entropies)

            # analyse
            ms = ee.measures
            measures.append(list(ms.values()))
            raw.append(ms)
        
        # numpy'd
        self.ees = ees
        self.titles = list(ees[0].measures.keys())[:5]
        self.entropies = np.array(entropies)
        self.ensembles, self.ergodics, self.divergences,\
            self.complexities, self.div_complexities \
                = np.stack(np.array(measures), axis=1)
        self.raw = raw

    def dataframe(self):
        import pandas as pd
        df = pd.DataFrame(data=self.raw)
        df[self.x_label] = self.x
        return df

    def bin_stats(self):
        return "%s bins from %s to %s" % (len(self.bins)-1, self.bins[0], self.bins[-1])

    def print_stats(self):
        df = self.dataframe()
        print("Measures at final %s=%s" % (self.x_label, self.x[-1]))
        print(df.iloc[-1])
        print("\n\nMeans")
        print(df.mean())
        print("\n\nMax's")
        print(df.max())

    def plot(self):
        """ Plots the evolution of the entropies & complexities over time """
        import seaborn as sns
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(15,5))
        
        #
        # First plot
        #

        # ensembles
        ensemble_entropies = np.stack(self.entropies, axis=1)
        colors = ensemble_entropies.shape[0]
        palette = sns.color_palette("mako_r", colors)
        alpha = 2.0/colors if 2.0/colors < 0.9 else 0.9
        for i, e in enumerate(ensemble_entropies):
            sns.lineplot(x=self.x, y=e, alpha=alpha, ax=axes[0], color=palette[i])

        # ergodic
        g = sns.lineplot(x=self.x, y=self.ergodics, ax=axes[0],
            label="Ergodic entropy", color="red")

        # ensemble mean
        g = sns.lineplot(x=self.x, y=self.ensembles, ax=axes[0],
            label="Mean ensemble Entropy", color="orange")

        g.set(ylim=(0,None))
        g.set_xlabel(self.x_label)
        g.set_ylabel("Entropy")
        g.set_title("Entropies (incl semi-transparent individual ensembles)")
        

        #
        # Second plot
        #
        h = sns.lineplot(x=self.x, y=self.divergences, ax=axes[1],
            color="orange", label="Ergodic divergence")
        h = sns.lineplot(x=self.x, y=self.div_complexities, ax=axes[1],
            color="skyblue", label="Ergodic Complexity (1st moment)")
        h = sns.lineplot(x=self.x, y=self.complexities, ax=axes[1],
            color="slateblue", label="Ergodic Complexity (2nd moment)")

        h.set(ylim=(0,1))
        h.set_xlabel(self.x_label)
        h.set_title(self.title if self.title else "Evolution of ergodic complexity and divergence")
        
        return fig

    def step_plot(self, index, comment=""):
        title = "%s=%s, step=%s (%s)" % (self.x_label, self.x[index], index, comment)
        self.ees[index].plot(title)

    def results(self, steps=False):
        """ Outputs most of the really interesting analysis """
        self.plot()
        
        if steps:
            # first results
            self.step_plot(0, "first")
            # first maximum plot
            max_pos = np.where(self.complexities == np.amax(self.complexities))[0][0]
            self.step_plot(max_pos, "max complexity")
        
            # last results
            self.step_plot(-1, "last")        

