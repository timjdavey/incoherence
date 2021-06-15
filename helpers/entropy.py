import numpy as np

def shannon_entropy(pmf, normalise=False):
    """
    Calculates the Shannon entropy for a
    discrete probability mass function.

    Discards all zero probabilty states.
    """
    pmf = np.array(pmf)

    if normalise:
        pmf = pmf/pmf.sum()
    
    # still need to run check for empty pdfs
    # assert_almost to deal with float point errors
    try:
        np.testing.assert_almost_equal(pmf.sum(), 1.0)
    except AssertionError:
        raise ValueError("pmf %s=%s is not normalised" % (pmf,pmf.sum()))

    # discard anything of zero probability
    pmf = np.ma.masked_equal(pmf,0).compressed()
    # could use nansum below, but rightly gives runtime warnings

    # treat as discrete entropy
    return -np.sum(pmf * np.log2(pmf))


def int_entropy(observations):
    """
    Work out entropy for a given set of observations
    Using a bin strategy of just int bounds
    """
    observations = np.array(observations)
    bins = np.arange(observations.max()+2)
    pdf, nbins = np.histogram(observations, bins=bins)
    return shannon_entropy(pdf, True)


class ErgodicEnsemble(object):
    """
    A simple model to help calculate the 

    Contains some simple performance boosts, but also stores
    some helpful data to make visualisation simpler
    (hence why it's a class rather than just a function).
    """
    def __init__(self, observations, bins):

        # Check data quality
        if len(observations.shape) != 2:
            raise ValueError("Need a set of observations, i.e. a 2d matrix,\
                but got %s" % len(observations.shape))
        if observations.shape[0] < 2:
            raise ValueError("Need at least 2 sets of observations")

        # Store
        self.observations = np.array(observations)
        self.ensembles_count = observations.shape[0]
        self.bins = bins

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

    def get_ergodic_observations(self):
        try:
            return self._ergodic_observations
        except AttributeError:
            self._ergodic_observations = self.observations.flatten()
            return self._ergodic_observations

    def get_entropies(self):
        try:
            return self._entropies
        except AttributeError:
            entropies = []
            for hist in self.get_histograms():
                entropies.append(shannon_entropy(hist, True))
            self._entropies = np.array(entropies)
            return self._entropies

    @property
    def ensemble(self):
        return np.mean(self.get_entropies())

    @property
    def geometric_ensemble_mean(self):
        return self.get_entropies().prod()**(1.0/len(self.get_entropies()))

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
        return self.ergodic - self.ensemble

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
                # Colours
                pal = sns.cubehelix_palette(self.ensembles_count, rot=-.25, light=0.7)

                # Initialize the FacetGrid object
                g = sns.FacetGrid(self.tidy_ensembles, row=tidy_variable,
                    hue=tidy_variable, aspect=8, height=0.5, palette=pal)
                
                # Draw the densities
                # Which aren't exactly the histograms, but a more visually clear rep
                g.map(sns.kdeplot, tidy_value,
                       clip_on=False,
                      fill=True, alpha=0.8, linewidth=0.5)
    
                # White plot outlines
                g.map(sns.kdeplot, tidy_value, clip_on=False, color="w", lw=1.0)
                # Set the subplots to overlap
                g.fig.subplots_adjust(hspace=-0.9)
                # Remove axes details that don't play well with overlap
                g.fig.suptitle('Distributions of each ensemble, as a ridge plot for clarity')
                g.set_titles("")
                g.set(yticks=[])
                g.despine(bottom=True, left=True)

            else:
                # Subplots
                fig, axes = plt.subplots(1, 2, sharey=True, figsize=(15,5))
                
                # Ensembles
                axes[0].set_title('Distributions by ensemble')
                # Colours - reverse colours
                pal = sns.cubehelix_palette(self.ensembles_count, rot=-.25, light=0.7, reverse=True)
                sns.kdeplot(ax=axes[0], data=self.tidy_ensembles, x=tidy_value, hue=tidy_variable,
                    # reverse order of plot, to match ridge
                    hue_order=[i for i in range(self.ensembles_count-1, -1, -1)],
                    fill=True, common_norm=False, palette=pal,
                    alpha=0.8, linewidth=0, legend=False)

                # Ergodic
                axes[1].set_title('Distribution of all observations (ergodic)')
                sns.kdeplot(ax=axes[1], data=self.tidy_ergo, x=tidy_value, hue=tidy_variable,
                    fill=True, common_norm=False, palette="flare",
                    alpha=1.0, linewidth=0, legend=False)

    def stats(self):
        msg = ""
        msg += "Ensembles count: %s\n" % self.ensembles_count
        msg += "Ergodic entropy: %.3f\n" % self.ergodic
        msg += "Average ensemble entropy: %.3f\n" % self.ensemble
        msg += "Ergodic Complexity: %.3f\n" % self.complexity
        print(msg)





