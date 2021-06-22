import numpy as np
import pandas as pd
import cellpylib as cpl

import seaborn as sns
from matplotlib import pyplot as plt


def complexity(ergodic, ensemble):
    """ This function might change, which is why it's encapsulated """
    return ensemble/ergodic

def diagnol_entropies(array, flip=False):
    """
    Calculates the entropy for diagnols
    """
    entropy = cpl.entropy.shannon_entropy

    steps, cells = array.shape

    if flip:
        array = np.fliplr(array)

    ents = []
    # upper triangle
    for i in range(cells):
        d = np.diag(array, i)
        ents.append(entropy([str(a) for a in d]))

    # lower triangle
    for i in range(1,steps):
        d = np.diag(array, -i)
        ents.append(entropy([str(a) for a in d]))

    return np.array(ents)


class CA1DEnsemble:
    """
    Simple model to help wrap cellpylib in a simple structure,
    for analysis & saving the data.

    In particular an Ensemble (a collection of )
    Celluar Automata that are 1 Dimensional.

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
    def __init__(self, rule=30, cells=200, count=1, init='random', folder='results/', p=0.9):
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
    
    @property
    def key(self):
        return "%s%s-%s-%s" % (self.folder, self.rule, self.cells, self.init)
    
    @property
    def analysis_keys(self):
        return ('ensemble', 'ergodic', 'meta')

    @property
    def timesteps(self):
        return len(self.raw[0])
    
    
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

    def analyse_ca(self, e):
        timesteps = len(e)
        stable = e[int(timesteps/2):]
        return {
             'Avg Cell Entropy' : cpl.average_cell_entropy(e),
             'Avg Stable Cell Entropy' : cpl.average_cell_entropy(stable),
             'Last Cell Entropy' : cpl.shannon_entropy(self.r2s(e[-1])),
             'Stable diag LR': diagnol_entropies(stable).mean(),
             'Stable diag RL': diagnol_entropies(stable, True).mean(),
             'Timesteps': timesteps,
             'Cells': self.cells,
             'Rule': self.rule,
             'Initial' : self.r2s(e[0]),
             'Init': self.init,
             'Ergodic': False,
            }

    def analyse(self):
        self.analyse_ensembles()
        self.analyse_ergodic()
        self.analysis = pd.DataFrame(self._raw_analysis)

    def analyse_ensembles(self):
        # check there is data to analyse
        if len(self.raw) == 0:
            raise Exception("No data. Please run .load() or .create() first.")
        
        analysis = []
        for e in self.raw:
            analysis.append(self.analyse_ca(e))
        self._raw_analysis = analysis
        return self._raw_analysis
    
    def analyse_ergodic(self):
        if self._raw_analysis is None:
            raise Exception("No data. Please .analyse_ensembles() first.")

        # for avg cell entropy
        whole = np.concatenate(self.raw)
        
        # for stable cell entropy
        timesteps = len(self.raw[0])
        half = np.concatenate([r[int(timesteps/2):] for r in self.raw])
        
        # last row
        last = np.concatenate([r[-1] for r in self.raw])
        
        # run analysis
        ergodic_analysis = {
            # Raw ergodic calcs
            'Avg Cell Entropy': cpl.average_cell_entropy(whole),
            'Avg Stable Cell Entropy': cpl.average_cell_entropy(half),
            'Last Cell Entropy': cpl.shannon_entropy(self.r2s(last)),
            'Stable diag LR': diagnol_entropies(half).mean(),
            'Stable diag RL': diagnol_entropies(half, True).mean(),
            # Freedom
            'Timesteps': timesteps,
            'Cells': self.cells,
            'Rule': self.rule,
            'Init': self.init,
            'Initial' : "",
            'Ergodic': True,
        }
        self._raw_analysis.append(ergodic_analysis)
        
    
    def r2s(self, row):
        return ''.join([str(s) for s in row])
    
    def plot_cpl(self, i=0):
        """ Plot using cpl """
        cpl.plot(self.raw[i])

    def plot_heatmap(self, ax=None):
        f = sns.heatmap(sum(self.raw), ax=ax,
            xticklabels=False, yticklabels=False)
        f.set_title("%s ensembles with rule %s" %
            (self.count, self.rule))


    def plot_single(self, i=None, ax=None):
        if i is None:
            i = np.random.randint(0, self.count)
        
        s = sns.heatmap(self.raw[i], ax=ax, cbar=False,
                xticklabels=False, yticklabels=False)
        s.set_title("Single plot of ensemble %s" % i)
        s.set_xlabel("%s Cells" % self.cells)
        s.set_ylabel("%s Timesteps" % self.timesteps)

    def plot_data(self, ax=None):

        # variables
        df = self.analysis
        m_first = 'Avg Cell Entropy'
        m_last = 'Stable diag RL'
        m = 'measurement'
        v = 'value'

        # all the cell means
        melt = pd.melt(df.loc[:self.count, m_first:m_last], var_name=m)
        sns.stripplot(x=m, y=v, data=melt, ax=ax,
                        alpha=0.3, zorder=1, palette="crest")
        
        # means of means
        sns.pointplot(y=v, x=m, ax=ax,
                      data=melt, palette="flare",
                      markers="v", scale=1, ci=None)
        
        # ergodic means
        erg = pd.melt(
            # ergodic value appended last
            # should probably get via Ergodic=True
            df.loc[self.count:self.count+1, m_first:m_last],
            var_name=m)
        g = sns.pointplot(y=v, x=m, ax=ax,
                      data=erg, palette="flare",
                      markers="^", scale=1, ci=None)
        
        # labels
        if ax is None:
            ax = g
        ax.set_xlabel('')
        ax.set_ylabel('Entropy value')
        ax.set_title("Ergodic & Mean entropy values")


    def plot(self, i=None):
        fig, axes = plt.subplots(1, 3,
            sharex=False, sharey=False, figsize=(20,5))
        self.plot_heatmap(axes[0])
        self.plot_single(ax=axes[1])
        self.plot_data(axes[2])
        



