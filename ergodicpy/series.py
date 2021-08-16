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
            # horrible mess of a transformation
            ny = np.stack(np.hstack(np.stack(self.y, axis=2)),axis=1)
            self.bins = binr(observations=ny)
        
        ees = []
        entropies = []
        measures = []
        
        # stored across all steps
        for timestep_obs in self.y:
            ee = ErgodicEnsemble(timestep_obs, self.bins)
            ees.append(ee)
            entropies.append(ee.entropies)
            measures.append(ee.measures[:4])
        
        # numpy'd
        self.ees = ees
        self.entropies = np.array(entropies)
        self.ensembles, self.ergodics, self.divergences, self.complexities = np.stack(np.array(measures), axis=1)
    
    @property
    def complexity_max(self):
        return self.complexities.max()

    @property
    def complexity_mean(self):
        return self.complexities.mean()

    @property
    def complexity_trend(self):
        trend = int(len(self.x)*0.1)
        return self.complexities[-trend:].mean()

    def stats(self):
        """ Prints out basic summary statistics """
        msg = ""
        msg += "%.1f%% maximum " % (self.complexity_max*100)
        msg += "%.1f%% mean " % (self.complexity_mean*100)
        msg += "%.1f%% trending " % (self.complexity_trend*100)
        print(msg)
        return msg
    
    def plot(self):
        """ Plots the evolution of the entropies & complexities over time """
        import seaborn as sns
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(15,5))
        
        # ensembles
        for e in np.stack(self.entropies, axis=1):
            sns.lineplot(x=self.x, y=e, alpha=0.15, ax=axes[0])

        # ensemble mean
        sns.lineplot(x=self.x, y=self.ensembles, ax=axes[0], label="Mean ensemble Entropy")
        
        # ergodic
        g = sns.lineplot(x=self.x, y=self.ergodics, ax=axes[0], label="Ergodic entropy")
        g.set(ylim=(0,None))
        g.set_xlabel(self.x_label)
        g.set_ylabel("Entropy")
        g.set_title("Entropies (incl semi-transparent individual ensembles)")
        
        # complexities
        ax2 = axes[1].twinx()
        h = sns.lineplot(x=self.x, y=self.complexities, ax=ax2, label="Ergodic complexity")
        h.set(ylim=(0,1))
        h.set_ylabel("Ergodic Complexity")

        # divergence
        f = sns.lineplot(x=self.x, y=self.divergences, ax=axes[1], color="orange", label="Ergodic divergence")
        f.set(ylim=(0,None))
        f.set_xlabel(self.x_label)
        f.set_ylabel("Ergodic Divergence")
        f.set_title(self.title if self.title else "Evolution of ergodic complexity and divergence")

        # legend box combined
        h1, l1 = axes[1].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        axes[1].legend(h1+h2, l1+l2, loc=2)
        ax2.get_legend().remove()
        
        return fig

    def report(self):
        self.stats()
        return self.plot()

