import pandas as pd
import numpy as np
import seaborn as sns
from .ergodic import ErgodicEnsemble

class ErgodicFeatures:

    def __init__(self, df, dist_name, ensemble_names, weighted=True):
        """
        For a given dataset & list of possible suitable ensembles,
        it creates a dict of ErgodicEnsemble's.
    
        :df: a dataframe of the data
        :dist_name: the distribution of interest e.g. 'house price'
        :ensemble_names: possible ensembles (typically columns in the data) e.g. ['region', 'year']
        :weighted: _True_ whether to weight the ergodic calculations by the ensemble frequency
        """
    
        ees = {}
        # for each of the candidate ensembles e.g. ['region', 'year']
        for candidate in ensemble_names:
            observations = []
            weights = []
    
            # loop through each of the ensembles in each candidate e.g. ['Uk', 'US'] in ['region']
            labels = df[candidate].unique()
            for r in labels:
                # filter out the observations
                vals = df.loc[df[candidate] == r][dist_name].to_numpy()
                # remove nan values
                vals = vals[np.logical_not(np.isnan(vals))]
                observations.append(vals)
                weights.append(len(vals))

            # allow override to turn off weights
            if not weighted:
                weights = None 

            # store the analyser class
            ee = ErgodicEnsemble(observations, weights=weights,
                labels=labels, ensemble_name=candidate, dist_name=dist_name)
            ees[candidate] = ee
        
        self.map = ees
        self.ensembles = ensemble_names

    def stats(self):
        """
        :returns: a dataframe of key statistics by each potential ensemble
        """
        columns = ['complexity', 'chi2', 'chi p']
        data = []
        for e in self.map.values():
            row = []
            row.append(e.complexity)
            chi2, p, _, _ = e.chi2()
            row.append(chi2)
            row.append(p)
            data.append(row)
        return pd.DataFrame(data=data, index=self.ensembles, columns=columns)

    def dependant(self, complexity=0.1, chi_p=0.05, split=False):
        """
        :returns: a list of ensembles which you should consider features
        """
        df = self.stats() # get fresh
        comps = list(df[(df['complexity'] >= complexity)].index)
        chips = list(df[(df['chi p'] < chi_p)].index)

        if split:
            return comps, chips
        else:
            return list(set(comps+chips))

    def independant(self, *args, **kwargs):
        """
        :returns: a list of ensembles which should _not_ be considered
        """
        return list(set(self.ensembles) - set(self.dependant(*args, **kwargs)))

    def plot(self):
        """
        Plots the ergodic graphs for each potential ensemble
        """
        for e in self.map.values():
            e.plot()
