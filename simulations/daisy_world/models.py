import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from functools import cached_property

import ergodicpy as ep

from .agent import Daisy
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid
from mesa.time import RandomActivation


POP_DEFAULT = {
    'white': {'albedo': 0.75, 'initial': 0.2},
    'black': {'albedo': 0.25, 'initial': 0.2},
}


class DaisyWorld(Model):
    """
    A Mesa port (with some expansions) of the James Lovelock DaisyWorld model.

    This port matches the NetLogo version
    http://www.netlogoweb.org/launch#http://www.netlogoweb.org/assets/modelslib/Sample%20Models/Biology/Daisyworld.nlogo
    
    Standard inputs
    :population: `dict` the population of species in the world e.g. see POP_DEFAULT
    :size: `int 29` the width in cells of the world (which is square i.e. `.area` is size**2)
    :luminosity: `float 1.0` is the luminosity of the sun, which can be reset throughout
    :surface_albedo: `float 0.4` albedo of an empty cell
    :store: `bool True` store common variables to `DataCollector` accessed via `.df` 

    Enhanced model inputs
    :mutate_p: `float 0.0` probability that a daisy will mutate on seeding
    :mutate_a: `float 0.05` the variance of the normal of much the albedo mutates by

    Basic Usage
    >> w = DaisyWorld(POP_DEFAULT) # initialise
    >> w.simulate(1000) # run simulation for 1000 steps
    >> w.plot() # plots all the graphs
    >> w.df # returns a dataframe of all the variables

    """
    
    def __init__(self, population,
            # standard temp variables
            size=29, luminosity=1.0, surface_albedo=0.4, 
            # mutate variables
            mutate_p=0.0,
            mutate_a=0.05,
            # varies max_age so doesn't boom / bust
            vary_age=False,
            # store data in datacollector or not
            store=True): 

        # setup grid world
        self.size = int(size)
        self.grid = SingleGrid(size, size, True)
        
        # temperature
        self.temperatures = np.zeros((size, size))
        self.luminosity = luminosity
        self.surface_albedo = surface_albedo

        # mutation
        self.mutate_p = mutate_p
        self.mutate_a = mutate_a

        # age
        self.vary_age = vary_age
        
        # keep counts consistent
        self.agents_ever = 0
        self.schedule = RandomActivation(self)
        
        # you'd want to turn off storing
        # for minor performance gains        
        if store:
            self.datacollector = DataCollector(
                model_reporters = {
                    'entropy':'entropy',
                    'temperature': 'temperature',
                    'luminosity': 'luminosity',
            })
        else:
            self.datacollector = None
        
        # populate the world
        # can be done after initialisation as well
        # although the counts won't be stored
        # adding the 'empty' cell "species"
        self.legend = {'empty': 0}
        if population:
            self.populate(population)
    
    """
    Populating the world
    """

    
    def populate(self, population):
        """
        Add a population to the world.
        
        :population: takes a dict of the form,
        
            daisies = {
                'white': {'albedo': 0.25, 'initial': 0.02},
                'black': {'albedo': 0.75, 'initial': 0.02},
            }
        
        where `initial` is the % coverage to add of that new population type
        """

        # for each in the population
        for name, traits in population.items():
            
            # add to the legend
            self.add_species(name, mutation=False)

            # create the given % of them
            how_many = int(self.area*traits['initial'])
            
            # then place randomly on the grid
            positions = self.random_cells(how_many)
            
            for pos in positions:
                self.add_agent(pos, name, traits['albedo'])

    def add_species(self, name, mutation=False):
        """ Adds a species to the legend & collector """
        
        # add to legend
        if name not in self.legend:
            self.legend[name] = len(self.legend)
            new_name = name
        
        # if it's a mutation it exists by definition
        # then it mutates the existing species
        elif mutation:
            # will become a nested set of names
            # tracked by id
            # parents would be new_name.split(">")
            l = len(self.legend)
            new_name = "%s>%s" % (name, l)
            self.legend[new_name] = l
        
        # if storing data to datacollector
        # need to add new column
        if self.datacollector is not None:
            self.datacollector.model_reporters[new_name] = lambda s: s.counts[new_name]
            # populate with zeroes for previous time steps (as didn't exist)
            self.datacollector.model_vars[new_name] = list(np.zeros(self.schedule.steps, dtype='int'))

        return new_name


    def add_agent(self, pos, name, albedo):
        """
        Adds an agent (Daisy)
        
        :pos: where they are to be added to (see random_cells())
        :name: type of daisy e.g. white or black
        :albedo: the albedo of the daisy
        """
        if self.vary_age:
            lifespan = 20 + np.random.randint(10)
        else:
            lifespan = 25
        
        agent = Daisy(self.agents_ever, self, name, albedo, lifespan)
        self.grid.place_agent(agent, pos)
        self.schedule.add(agent)
        self.agents_ever += 1


    def random_cells(self, amount=1):
        """
        Used in replacement for self.grid.position_agent()
        as that function is unreasonably slow.
        It uses sorted(self.grid.empties) which actually
        makes it less random and slower!
        """
        empties = list(self.grid.empties)
        idx = np.random.choice(len(empties), amount, replace=False)
        return [empties[i] for i in idx]

    """
    Running the model
    """

    def step(self, observations=False):
        """ Move the model one step on """
        # adjust the temperature
        self._update_global_temperature()
        
        # step all the agents
        # spawning or dieing the daisies
        self.schedule.step()
        
        # store variables
        self._collect()

        # optionally return obs on each step
        if observations:
            return self.observations()
        
        
    def simulate(self, ticks):
        for i in range(ticks):
            self.step()
    
    """
    Temperature calculations
    """

    def _calculate_temperature(self, albedo, current_temperature):
        """ Calculates new temperature of a cell
        
        :albedo: given albedo of what's in the cell
        :current_temperature:
        """
        absorbed_luminosity = (1 - albedo) * self.luminosity
    
        # calculate the local heating effect
        if absorbed_luminosity > 0:
            local_heating = 72 * np.log(absorbed_luminosity) + 80
        else:
            local_heating = 80
    
        # set the temperature at to be the average of the current temperature
        # and the local-heating effect
        return (current_temperature + local_heating) / 2
    
    def _update_temperature(self, albedo, pos):
        """ Updates the albedo of a given cell"""
        x, y = pos
        t = self._calculate_temperature(albedo, self.temperatures[x][y])
        self.temperatures[x][y] = t
    
    def _update_global_temperature(self):
        """
        Adjust for local temperatures within cells
        based on the temperatures of their local neighbourhood.

        Ported from the excellent
        https://github.com/mchozhang/DaisyWorld
        """
        
        # update agents cells
        for a in self.schedule.agents:
            self._update_temperature(a.albedo, a.pos)
        
        # update empty cells
        for pos in self.grid.empties:
            self._update_temperature(self.surface_albedo, pos)
        
        # would rather update all temperatures at once
        # using a temporary temperature grid
        # however, netlogo does a rolling average
        # and consistency is more important
        for cx in range(self.size):
            for cy in range(self.size):
                
                # average over neighbours
                absorbed = 0
                for nx, ny in self.grid.iter_neighborhood((cx,cy), moore=True):
                     absorbed += self.temperatures[nx][ny]
                absorbed /= 8
                
                # calculate ultimate temperature after diffusion
                self.temperatures[cx][cy] = (self.temperatures[cx][cy] + absorbed) / 2

    """
    Helpful data structures
    """
    
    @cached_property
    def area(self):
        return int(self.size**2)
    
    @cached_property
    def df(self):
        """ A dataframe of the current data """
        return self.datacollector.get_model_vars_dataframe()

    def grid_as_numpy(self, offset=0):
        """ A 2D numpy array of the world, where the int number
        in each cell is the population in that cell.
        Use `.legend` to see what population element corresponds to each number.
        """
        # initialise to empties
        data = np.zeros((self.size, self.size), dtype='uint8')
        
        # for each of the agents
        for a in self.schedule.agents:
            x, y = a.pos
            data[x][y] = a.legend+offset

        return data
    
    def observations(self):
        return self.grid_as_numpy().flatten()

    @property
    def histogram(self):
        """ Returns a histogram of the current counts """
        return list(self.counts.values())

    """
    Key stats to store
    """
    
    STATS_TO_CLEAR = ['counts', 'entropy', 'temperature', 'df']
    
    def _collect(self):
        """ Collect the data at the current step """
        
        # only store if collecting
        if self.datacollector is not None:
            # clear the cached variables
            for stat in self.STATS_TO_CLEAR:
                try:
                    delattr(self, stat)
                except AttributeError:
                    # ignore if hasn't yet cached
                    pass
            
            # store the data
            self.datacollector.collect(self)

    @cached_property
    def counts(self):
        """ Returns a dict of the counts of each population type (incl empty) """
        counts = dict([(k, 0) for k in self.legend.keys()])
        for a in self.schedule.agents:
            counts[a.name] += 1
        counts['empty'] = len(self.grid.empties) # alt is self.area-sum(counts)
        return counts
    
    @cached_property
    def entropy(self):
        """ Returns current entropy of the world """
        return ep.shannon_entropy(self.histogram, True)
    
    @cached_property
    def temperature(self):
        """ Returns the mean temperature of the world """
        return self.temperatures.mean()
    
    """
    Perturbation
    """
    def plague(self, percent=0.5, name=None):
        """
        Causes a sudden number of agents to die
        
        :percent: 0.5 percentage of that type which should die
        :name: optionally if only want a certain type to die
        """
        agents = self.schedule.agents

        # filter if type is supplied
        if name:
            agents = list(filter(lambda a: a.name == name, agents))

        num_to_die = int(len(agents)*percent)
        
        # then kill them
        for a in np.random.choice(agents, num_to_die, replace=False):
            a.die()


    """
    Plots
    """
    
    def plot(self):
        """ Plots all figures """
        fig, axes = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(15,10))
        self.plot_grid(axes[0][0])
        self.plot_entropy(axes[0][1])
        self.plot_temperature(axes[1][0])
        self.plot_global_temp(axes[1][1])
    
    def plot_grid(self, ax=None):
        """ Plots a heatmap visual of each of the populations on the world """
        offset = 1
        data = self.grid_as_numpy(offset)
        cmap = sns.color_palette("cubehelix_r", as_cmap=True)
        vmax = len(self.legend)+1
        g = sns.heatmap(data, cmap=cmap, ax=ax,
                xticklabels=False, yticklabels=False, vmin=0, vmax=vmax)
        return g
    
    def plot_temperature(self, ax=None):
        """ Plots the temperature of each cell at this point in time """
        g = sns.heatmap(self.temperatures, ax=ax,
                   xticklabels=False, yticklabels=False)
        return g
    
    def plot_global_temp(self, ax=None):
        """ Plots how the global temperature has changed over time """
        d = self.df.temperature
        g = sns.lineplot(data=d, ax=ax)
        return g
    
    def plot_entropy(self, ax=None):
        """ Plots how the entropy has changed over time """
        e = self.df.entropy
        g = sns.lineplot(data=e, ax=ax)
        g.set(ylim=(0, None))
        return g

