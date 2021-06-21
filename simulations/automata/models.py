import numpy as np
import pandas as pd
import cellpylib as cpl


def complexity(ergodic, ensemble):
    """ This function might change, which is why it's encapsulated """
    return ensemble/ergodic


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
        self.p = p
        self.folder = folder
        self._raw_analysis = None
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
            print("Completed ", i)
        
        self.raw = new_raw
    
    @property
    def key(self):
        return "%s%s-%s-%s" % (self.folder, self.rule, self.cells, self.init)
    
    @property
    def analysis_keys(self):
        return ('ensemble', 'ergodic', 'meta')
    
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
            print("Ensemble analysis not found. Please run .analyse_ensemble()")

    def analyse_ca(self, e):
        timesteps = len(e)
        return {
             'Avg Cell Entropy' : cpl.average_cell_entropy(e),
             'Avg Stable Cell Entropy' : cpl.average_cell_entropy(e[int(timesteps/2):]),
             'Last Cell Entropy' : cpl.shannon_entropy(self.r2s(e[-1])),
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
    
    def plot(self, i=0):
        cpl.plot(self.raw[i])