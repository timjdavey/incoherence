from .discrete import Discrete
from .bins import binint


class Collection(Discrete):
    """
    Wrapper around Discrete.

    So you can create an Ensembles directly from
    histogram data (i.e. probability distributions)
    rather than raw observations.

    :histograms: list of histogram data e.g. [[1,2], [0,12]]
    :bins: list of bin values for the histograms
    """

    def __init__(self, histograms, bins=None, **kwargs):
        lengths = set([len(h) for h in histograms])
        if len(lengths) > 1:
            raise ValueError("histogram lengths not equal %s" % lengths)

        if bins is None:
            # default to incremental numbering
            bins = binint(0, len(histograms[0]))
        super().__init__(histograms, bins, histograms=True, **kwargs)

    def bayesian_observations(self, *args, **kwargs):
        raise NotImplementedError
