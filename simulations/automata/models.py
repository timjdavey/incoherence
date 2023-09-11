import numpy as np
import pandas as pd
import cellpylib as cpl
import seaborn as sns
import matplotlib.pyplot as plt

import ensemblepy as ep
from ensemblepy.stats import _incoherence

from cellpylib.entropy import shannon_entropy


def r2e(row):
    """ 'row 2 entropy'. CA row (array) of 1's & 0's
    
    :returns: entropy float.
    """
    return shannon_entropy([str(s) for s in row])

def diagnol_entropies(array, flip=False):
    """
    Calculates the entropy for diagnols of a CA.

    :flip: default is top left to bottom right,
    if flipped, it goes top right to bottom left.

    :returns: an array of entropy floats.
    """
    steps, cells = array.shape

    if flip:
        array = np.fliplr(array)

    ents = []
    # upper triangle
    for i in range(cells):
        d = np.diag(array, i)
        ents.append(r2e(d))

    # lower triangle
    for i in range(1,steps):
        d = np.diag(array, -i)
        ents.append(r2e(d))

    return np.array(ents)


class CA1DEnsemble:
    """
    Simple model to help wrap cellpylib in a simple structure,
    for analysis & saving the data.

    In particular an Ensemble (a collection of )
    Celluar Automata that are 1 Dimensional.

    :rule: the wolfram new kind of science rule number
    :cells: the number of cells
    :count: the number of ensembles
    :init: takes {random|simple|half|sparse}

    There's a very simple flow to it's usage being e.g.
    ca = CA1DEnsemble(rule=30, cells=200, count=5, init='random')
    
    # run each ensemble how many timesteps
    # save as it goes
    ca.run(1000, save=True)
    # or call save
    ca.save()
    
    # run the analysis on all the data
    ca.analyse()

    # save the analysis
    ca.save()
    # save saves everything it can


    # To load the data
    # Initialise an object with the variables already run
    # then call .load()
    # helpful between sessions & if the analysis function is updated
    cb = CA1DEnsemble(rule=30, cells=200, init='random')
    cb.load()

    # refresh analysis
    cb.analyse()

    """
    def __init__(self, rule=30, cells=200, count=1,
            init='random', folder='results/', p=0.9,
            power=0.5, k=None):
        # specify major rules
        self.rule = rule
        self.cells = cells
        self.init = init
        self.count = count
        self.p = p
        self.folder = folder
        self._raw_analysis = None
        self.analysis = None
        self.raw = []
        self.power = power
        self.k = k
        
        # even if loading them from a file a single creation won't hurt
        self.create(count)
    
    def create(self, ensembles):
        if self.init == 'random':
            for i in range(ensembles):
                self.raw.append(cpl.init_random(self.cells))
        elif self.init == 'simple':
            for i in range(ensembles):
                self.raw.append(cpl.init_simple(self.cells))
        elif self.init == 'half':
            t2 = int(self.cells/2)
            for i in range(ensembles):
                self.raw.append(
                    np.array([
                        np.append(np.ones(t2, dtype=int), np.zeros(t2, dtype=int))
                        ]))
        elif self.init == 'sparse':
            for i in range(ensembles):
                self.raw.append(
                    np.array([
                        np.random.choice([0,1],self.cells,p=[self.p,1-self.p])
                    ]))
    
    """
    Execution

    """


    def run(self, distance, save=False):
        new_raw = []
        for i, e in enumerate(self.raw):
            e = cpl.evolve(e, timesteps=distance, 
                apply_rule=lambda n, c, t: cpl.nks_rule(n, self.rule))
            new_raw.append(e)
            
            # save as it progresses
            if save:
                self.save(new_raw)
        
        self.raw = new_raw
    
    def run_or_load(self, distance, save=False):
        try:
            self.load()
        except FileNotFound:
            self.run(distance, save)
    


    """
    Saving & loading data

    """

    @property
    def key(self):
        return "%s%s-%s-%s" % (self.folder, self.rule, self.cells, self.init)
    
    def save(self, delta=None):
        # optionally save current status
        if delta is None:
            delta = self.raw
        np.savez(self.key, delta)
        
        # save the different analysis
        if self.analysis is not None:
            self.analysis.to_csv("%s.csv" % self.key)
    
    def load(self):
        # load raw data
        self.raw = np.load("%s.npz" % self.key)['arr_0']
        
        # load pandas dataframe of analysis
        try:
            self.analysis = pd.read_csv("%s.csv" % self.key, index_col=0)
        except FileNotFound:
            raise FileNotFound("Ensemble analysis not found. Please run .analyse_ensemble()")


    """
    Analysis

    """

    def analyse(self):
        self._analyse_ensembles()
        self._analyse_ergodic()
        self.analysis = pd.DataFrame(self._raw_analysis)
        self._analyse_complexity()
        self.analysis = pd.DataFrame(self._raw_analysis)

    def _dv(self, stable=None):
        """Density variance entropy estimator
        Turns array into a list of positions for density variance"""

        if self.k is None: return 0

        stable_steps = len(self.raw[0])//2

        if stable is None:
            # then doing it for ergodic
            positions = []
            for e in self.raw:
                # identifier is the more common
                iden = 1 if np.sum(e) < e.size/2 else 0
                positions += list(zip(*np.where(e[stable_steps:]==iden)))
        else:
            iden = 1 if np.sum(stable) < stable.size/2 else 0
            positions = list(zip(*np.where(stable==iden)))

        if len(positions) > 1:
            # normalise the positions
            positions = np.array(positions, 'float64')
            positions[:, 1] /= float(self.cells)
            positions[:, 0] /= float(stable_steps)
            return ep.density_variance(positions, power=self.power, k=self.k)
        else:
            # if no positions, then high entropy
            return 1

    def _stable(self):
        return [e[len(e)//2:] for e in self.raw]


    def _analyse_ensembles(self):        
        rows = []
        for stable in self._stable():
            rows.append({
             #'Avg Cell Entropy' : cpl.average_cell_entropy(e),
             'Avg Stable Cell Entropy' : cpl.average_cell_entropy(stable),
             #'Last Cell Entropy' : r2e(e[-1]),
             'Stable diag LR': diagnol_entropies(stable).mean(),
             'Stable diag RL': diagnol_entropies(stable, True).mean(),
             'Stable density variance': self._dv(stable),
             #'Initial' : "".join([str(x) for x in stable[0]]),
             'Kind': 'ensemble',
            })
        self._raw_analysis = rows
        return self._raw_analysis
    
    def _analyse_ergodic(self):
        # for avg cell entropy
        #whole = np.concatenate(self.raw)

        # last row
        #last = np.concatenate([r[-1] for r in self.raw])
        
        # for stable cell entropy
        stable = self._stable()
        half = np.concatenate(stable)

        
        # run analysis
        ergodic_analysis = {
            # Raw ergodic calcs
            #'Avg Cell Entropy': cpl.average_cell_entropy(whole),
            'Avg Stable Cell Entropy': cpl.average_cell_entropy(half),
            #'Last Cell Entropy': r2e(last),
            'Stable diag LR': diagnol_entropies(half).mean(),
            'Stable diag RL': diagnol_entropies(half, True).mean(),
            'Stable density variance': self._dv(),
            'Initial' : "",
            'Kind': 'ergodic',
        }
        self._raw_analysis.append(ergodic_analysis)
    
    def get_analysis(self, kind='ensemble'):
        return self.analysis.loc[self.analysis['Kind']==kind].loc[:,self.analysis.columns[0:-2]]

    def _analyse_complexity(self):

        # Store complexities
        data = {}
        data['Initial'] = ""
        data['Kind'] = "complexity"
    
        ensemble_data = self.get_analysis("ensemble")
        ergodic_data = self.get_analysis("ergodic")

        for key in ensemble_data:
            data[key] = _incoherence(list(ergodic_data[key])[0], list(ensemble_data[key]))
        
        self._raw_analysis.append(data)

    @property
    def incoherence(self):
        return self.get_analysis('complexity')['Avg Stable Cell Entropy'].to_list()[0]


    """
    Plotting

    """

    def plot(self, i=0, j=None, inc=None):
        col = 2 if j is None else 3
        fig, axes = plt.subplots(1, col,
            sharex=False, sharey=False, figsize=(4*col,4))
        plt.tight_layout()
        self.plot_heatmap(axes[0], inc=inc)
        self.plot_single(i, ax=axes[1])
        if j is not None:
            self.plot_single(j, ax=axes[2])
        return fig

    def plot_cpl(self, i=0):
        """ Plot using cpl """
        cpl.plot(self.raw[i])

    def plot_heatmap(self, ax=None, inc=None):
        f = sns.heatmap(sum(self.raw), ax=ax, cbar=False,
            xticklabels=False, yticklabels=False, square=True)
        f.set_title("Rule %s with incoherence of %.2f" %
            (self.rule, self.incoherence if inc is None else inc))
        return f

    def plot_single(self, i=0, ax=None):
        s = sns.heatmap(self.raw[i], ax=ax, cbar=False,
                xticklabels=False, yticklabels=False, square=True)
        s.set_title("Single sample out of %s" % self.count)
        s.set_xlabel("%s Cells" % self.cells)
        s.set_ylabel("%s Timesteps" % len(self.raw[i]))
        return s

    def plot_data(self, ax=None):

        # variables
        df = self.analysis
        m = 'measurement'
        v = 'value'

        # all the cell means
        melt = pd.melt(self.get_analysis('ensemble'), var_name=m)
        sns.stripplot(x=m, y=v, data=melt, ax=ax,
                        alpha=0.4, zorder=1, palette="crest")
        
        # means of means
        sns.pointplot(y=v, x=m, ax=ax,
                      data=melt, palette="flare",
                      markers="_", scale=1, ci=None)
        
        # ergodic means
        erg = pd.melt(self.get_analysis('ergodic'), var_name=m)
        g = sns.pointplot(y=v, x=m, ax=ax,
                      data=erg, palette="flare",
                      markers="_", scale=1, ci=None)
        
        # labels
        if ax is None:
            ax = g
        ax.set_xlabel('')
        ax.set_ylabel('Entropy value')
        ax.set_ylim([0,1])
        ax.set_title("Ergodic (higher) & Mean (lower) entropy values")
        
    def show(self, steps):
        self.run(steps)
        self.analyse()
        self.plot()
        return self


