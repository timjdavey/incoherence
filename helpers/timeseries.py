import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from helpers.ergodic import ErgodicEnsemble


class ErgodicTimeSeries:
    """
    Simple class to handle the ergodic analysis over a time series of data.
    It is designed to be extended e.g. see <link to DaisyWorld example>
    """
    def __init__(self, log_func=lambda x, *args, **kwargs: x):
        self.observations = []
        self.log = log_func
    
    def analyse(self):
        """
        Creates ErgodicEnsembles for each timestep
        and stores the analysis for plotting
        """
        self.log("Analysing")
        
        # data for each plot line
        ees = []
        entropies = []
        complexities = []
        ergodics = []
        
        # stored across all steps
        for timestep_obs in self.observations:
            
            ee = ErgodicEnsemble(timestep_obs, self.bins)
            ees.append(ee)
            entropies.append(ee.entropies)
            complexities.append(ee.complexity)
            ergodics.append(ee.ergodic)
        
        # numpy'd
        self.ees = ees
        self.entropies = np.array(entropies)
        self.complexities = np.array(complexities)
        self.ergodics = np.array(ergodics)
        self.timesteps = range(len(self.observations))
    
    @property
    def complexity_max(self):
        return self.complexities.max()

    @property
    def complexity_mean(self):
        return self.complexities.mean()

    @property
    def complexity_trend(self):
        trend = int(len(self.timesteps)*0.1)
        return self.complexities[-trend:].mean()

    def stats(self):
        """ Prints out basic summary statistics """
        msg = ""
        msg += "%.1f%% maximum " % (self.complexity_max*100)
        msg += "%.1f%% mean " % (self.complexity_mean*100)
        msg += "%.1f%% trending " % (self.complexity_trend*100)
        self.log(msg)
        return msg
    
    def plot(self):
        """ Plots the evolution of the entropies & complexities over time """
        self.log("Plotting")
        
        fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(15,5))
        
        # ergodics
        for e in np.stack(self.entropies, axis=1):
            sns.lineplot(x=self.timesteps, y=e, alpha=0.15, ax=axes[0])
        means = self.entropies.mean(axis=1)
        sns.lineplot(x=self.timesteps, y=means, ax=axes[0], label="Ensemble Mean")
        g = sns.lineplot(x=self.timesteps, y=self.ergodics, ax=axes[0], label="Ergodic")
        g.set(ylim=(0,None))
        g.set_xlabel("Timesteps")
        g.set_ylabel("Entropy of system")
        g.set_title("Evolution of entropies")
        
        # complexities
        g = sns.lineplot(x=self.timesteps, y=self.complexities, ax=axes[1])
        g.set(ylim=(0,1))
        g.set_xlabel("Timesteps")
        g.set_ylabel("Ergodic Complexity of system")
        g.set_title("Evolution of ergodic complexity")
    
    def complete(self, distance=500):
        """ Does all the standard steps for a given distance """
        self.simulate(distance)
        self.analyse()
        self.plot()
        self.stats()

