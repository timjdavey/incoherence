from .model import DaisyWorld

from helpers.timeseries import ErgodicTimeSeries



class DaisyWorldAnalysis(ErgodicTimeSeries):
    """
    Simple class to handle the ergodic analysis of DaisyWorld
    """
    def __init__(self, worlds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worlds = worlds
    
    def simulate(self, distance=100):
        """ Runs the simulation for all worlds for `distance` steps """
        timestep_obs = []
        species_max = 0
        
        for d in range(distance):
            self.log("Simulating", d)
            ensemble_obs = []
            for w in self.worlds:
                w.step()
                species_max = max(species_max, len(w.legend))
                ensemble_obs.append(w.grid_as_numpy().flatten())
            timestep_obs.append(ensemble_obs)
        
        self.observations += timestep_obs
        self.bins = list(range(species_max+1))