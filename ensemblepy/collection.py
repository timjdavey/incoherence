from .discrete import Discrete


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
        self.histograms = histograms

        if bins is None:
            # default to incremental numbering
            bins = binint(0,len(histograms[0]))
        observations = [histogram_to_observations(h, bins[:-1]) for h in histograms]
        
        super().__init__(observations, bins, **kwargs)

    def bayesian_observations(self, *args, **kwargs):
        raise NotImplementedError

