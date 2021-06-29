import pandas as pd
import numpy as np
import seaborn as sns
from .ergodic import ErgodicEnsemble


def complexity_stablise(bin_range, plot=False, *args, **kwargs):
    """
    Runs `ergodic_collection` for series of different bin numbers.
    To that you can get a sense check that you've binned correctly.
    Typically complexity values stablise at a certain level,
    but that level depends on the sparseness of the data.

    :bin_range: this is not the bins themselves, but the number of bins within the range
    e.g. range(5,10,50)
    :plot: whether to automatically plot the returns results
    :*args, **kwargs: passed to `ergodic_collection`

    :returns: a dataframe with the core metrics for each bin number
    """
    complexities = []
    
    # loop through bin_range
    for i in bin_range:
        kwargs['bin_number'] = i

        # create a collection for each bin type
        ees = ergodic_collection(*args, **kwargs)

        # store in dict ready for a dataframe
        store = {'bins': str(i)}
        for e in ees.values():
            store[e.ensemble_name] = e.complexity
        complexities.append(store)

    df = pd.DataFrame(complexities)
    if plot:
        melt = pd.melt(df, id_vars='bins')
        sns.lineplot(data=melt, x='bins', y='value', hue='variable')
    return df



def ergodic_collection(df, dist_name, ensemble_names, bin_number=20, display=False):
    """
    For a given dataset & list of possible suitable ensembles,
    it creates a dict of ErgodicEnsemble's.

    :df: a dataframe of the data
    :dist_name: the distribution of interest e.g. 'house price'
    :ensemble_names: possible ensembles (typically columns in the data) e.g. ['region', 'year']
    :bin_number: the number of bins to cut the values into
    :display: print out & plot the data for each of the ensembles as they're created

    :returns: a dict with candidate ensembles as keys and it's ErgodicEnsemble as values.
    """
    # create a simple bin structure
    bins = np.linspace(df[dist_name].min(), df[dist_name].max(), bin_number)

    ees = {}
    # for each of the candidate ensembles e.g. ['region', 'year']
    for candidate in ensemble_names:
        observations = {}

        # loop through each of the ensembles in each candidate e.g. ['Uk', 'US'] in ['region']
        for r in df[candidate].unique():
            # filter out the observations
            vals = np.concatenate(df.loc[df[candidate] == r].loc[:,[dist_name]].to_numpy())
            observations[r] = vals

        # store the analyser class
        ee = ErgodicEnsemble(observations, bins, candidate, dist_name)
        ees[candidate] = ee

        if display:
            ee.stats()
            ee.plot()
    return ees
