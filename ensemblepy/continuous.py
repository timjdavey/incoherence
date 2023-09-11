import numpy as np
from .stats import measures


class Continuous:
    """
    A base model to help calculate the various measures.

    Contains some simple performance boosts, but also stores
    some helpful data to make visualisation simpler
    (hence why it's a class rather than just a function).

    intialization
    :observations: either takes a list observations grouped by ensemble
    :weights: _None_, list, for the ensembles
    :labels: the names of the ensembles for plotting
    :lazy: _False_ will calculate measures on creation if False, otherwise need to call `analyse()`
    :normalise: _None_, normalises y data for continuous entropy calc
        defaults when None to (y.min(), y.max())
    """

    def __init__(
        self,
        observations,
        normalise=None,
        weights=None,
        labels=None,
        metrics=None,
        ensemble_name="ensemble",
        dist_name="value",
        lazy=False,
    ):
        self.discrete = False
        self.observations = list(observations)
        self.weights = weights
        self.labels = labels
        self.metrics = metrics

        pooled = np.concatenate(self.observations)
        if normalise is None:
            normalise = (pooled.min(), pooled.max())

        self.normalise = normalise

        # often ragged list, so need to do individually
        self.normalised = []
        for o in self.observations:
            data = np.array(o)
            if len(data) > 0:
                data = data[~np.isnan(data)]  # remove NaNs
                self.normalised.append(
                    (data - normalise[0]) / (normalise[1] - normalise[0])
                )

        if not lazy:
            self.analyse()

    def _get_measures(self):
        return measures(
            self.normalised, discrete=False, weights=self.weights, metrics=self.metrics
        )

    def analyse(self):
        """
        Gets measures and assigns values as attributes
        """

        # get measures
        ms = self._get_measures()

        for k, v in ms.items():
            setattr(self, k, v)

        self.measures = ms
        return ms

    def comparison(self):
        from scipy.stats import kruskal, f_oneway

        results = {"incoherence": self.incoherence}
        for name, func in (("Kruskal", kruskal), ("ANOVA", f_oneway)):
            v, p = func(*self.normalised)
            results[name] = v
            results["%s p" % name] = p

        results["std(means)"] = np.std(np.mean(self.observations, axis=1))
        results["std(stds)"] = np.mean(np.std(self.observations, axis=1))
        return results
