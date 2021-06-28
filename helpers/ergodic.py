import numpy as np

from .entropy import shannon_entropy, complexity


class ErgodicEnsemble:
    """
    A simple model to help calculate the 

    Contains some simple performance boosts, but also stores
    some helpful data to make visualisation simpler
    (hence why it's a class rather than just a function).
    """
    def __init__(self, observations, bins):

        self.observations = np.array(observations)
        self.ensembles_count = len(observations.shape)
        self.bins = bins

    """
    Simple data manipulation methods

    """

    def get_histograms(self):
        try:
            return self.histograms
        except AttributeError:
            histograms = []
            for obs in self.observations:
                hist, nbins = np.histogram(obs, bins=self.bins)
                histograms.append(hist)
            self.histograms = np.array(histograms)
            return self.histograms

    def get_entropies(self):
        try:
            return self._entropies
        except AttributeError:
            entropies = []
            for hist in self.get_histograms():
                entropies.append(shannon_entropy(hist, True))
            self._entropies = np.array(entropies)
            return self._entropies

    def get_ergodic_observations(self):
        return np.concatenate(self.observations)

    """
    Calculations & metrics
    
    """

    @property
    def ensemble(self):
        return np.mean(self.get_entropies())

    @property
    def geometric_ensemble_mean(self):
        ents = self.get_entropies()
        return ents.prod()**(1.0/len(ents))

    @property
    def ergodic(self):
        try:
            return self._ergodic
        except AttributeError:
            hist, nbins = np.histogram(self.get_ergodic_observations(), bins=self.bins)
            self._ergodic = shannon_entropy(hist, True)
            return self._ergodic

    @property
    def complexity(self):
        return complexity(self.ensemble, self.ergodic)



    """
    Plot
    
    """

    def plot(self, ridge=False):
        import warnings
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # no background
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        # Clean the data
        tidy_variable = 'variable'
        tidy_value = 'value'
        tidy_h = 'h'
        palette = 'flare'

        try:
            self.tidy_ensembles
        except AttributeError:
            self.tidy_ensembles = pd.melt(pd.DataFrame(self.observations).transpose())
            self.tidy_ergo = pd.DataFrame({
                tidy_variable:tidy_h,tidy_value:self.get_ergodic_observations()})
        
        # Ignore Tight layout warning
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)

            if ridge:

                # Initialize the FacetGrid object
                g = sns.FacetGrid(self.tidy_ensembles, row=tidy_variable,
                    hue=tidy_variable, aspect=8, height=0.5, palette=palette)
                
                # Draw the densities
                # Which aren't exactly the histograms, but a more visually clear rep
                g.map(sns.histplot, tidy_value,
                    element='step',
                    fill=True, alpha=0.8)
    
                # Set the subplots to overlap
                g.fig.subplots_adjust(hspace=-0.6)
                # Remove axes details that don't play well with overlap
                g.fig.suptitle('Distributions of each ensemble, as a ridge plot for clarity')
                g.set_titles("")
                g.set(yticks=[])
                g.despine(bottom=True, left=True)

            else:
                # Subplots
                fig, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(15,5))
                
                # Ensembles
                axes[0].set_title('Distributions by ensemble')
                # Colours - reverse colours
                sns.histplot(ax=axes[0],
                    bins=self.bins, element='step', fill=True,
                    data=self.tidy_ensembles, x=tidy_value, hue=tidy_variable,
                    palette="%s_r" % palette, alpha=0.3, legend=False)

                # Ergodic
                axes[1].set_title('Distribution of all observations (ergodic)')
                sns.histplot(ax=axes[1],
                    bins=self.bins, element='step',
                    data=self.tidy_ergo, x=tidy_value, hue=tidy_variable,
                    palette=palette, alpha=1.0, legend=False)

    def stats(self):
        msg = ""
        msg += "Ensembles count: %s\n" % self.ensembles_count
        msg += "Ergodic entropy: %.3f\n" % self.ergodic
        msg += "Average ensemble entropy: %.3f\n" % self.ensemble
        msg += "Ergodic Complexity: %.1f%%\n" % (self.complexity*100)
        print(msg)
