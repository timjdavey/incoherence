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
    def __init__(self, observations, normalise=None, weights=None,
                labels=None, ensemble_name='ensemble', dist_name='value',
                lazy=False):
        
        self.discrete = False
        self.observations = observations
        self.weights = weights
        self.labels = labels

        pooled = np.concatenate(observations)
        if normalise is None:
            normalise = (pooled.min(), pooled.max())

        # often ragged list, so need to do individually
        self.normalised = [(np.array(o) - normalise[0])/(normalise[1]-normalise[0]) for o in observations]

        if not lazy:
            self.analyse()

    def get_measures(self):
        return measures(self.normalised, discrete=False,
            weights=self.weights, with_meta=True)

    def analyse(self):
        """
        Gets measures and assigns values as attributes
        """

        # get measures
        ms = self.get_measures()

        for k, v in ms.items():
            setattr(self, k, v)

        del ms['entropies']
        del ms['weights']

        self.measures = ms
        return ms

