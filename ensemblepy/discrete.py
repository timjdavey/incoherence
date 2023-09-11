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

    def __init__(
        self,
        observations,
        bins,
        weights=None,
        metrics=None,
        labels=None,
        base=None,
        lazy=False,
        histograms=False,
    ):
        self.discrete = True
        self.observations = observations
        self.bins = bins
        self.weights = weights
        self.base = base
        self.labels = labels
        self.metrics = None
        if histograms:
            self.histograms = observations
        else:
            self.histograms = None

        # do analysis
        if not lazy:
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
        return measures(
            self.histograms,
            weights=self.weights,
            base=self.base,
            discrete=True,
            metrics=self.metrics,
        )

    def chi2(self, ignore=False):
        from scipy.stats import chi2_contingency

        # returns (chi2, p, dof, expected)
        try:
            return chi2_contingency(self.histograms)
        except ValueError:
            return None, None, None, None

    def comparison(self):
        c, p, _, _ = self.chi2()
        return {
            "incoherence": self.incoherence,
            "chi2": c,
            "chi2 p": p,
        }

    def ergodic_pmf(self):
        """
        The ergodic ensemble (pmf as normalised).
        With standard weights.
        """
        return point_pmf(self.histograms, weights=self.weights)


# handling legacy
Ensembles = Discrete
