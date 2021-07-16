import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from functools import cached_property

from helpers.entropy import shannon_entropy

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
    
    def __init__(self, population, size=29, luminosity=1.0, surface_albedo=0.4):
        # setup grid world
        self.size = int(size)
        self.grid = SingleGrid(size, size, True)
        
        # temperature store
        self.temperatures = np.zeros((size, size))
        self.luminosity = luminosity
        self.surface_albedo = surface_albedo
        
        # keep counts consistent
        self.agents_ever = 0
        self.schedule = RandomActivation(self)
        
        # populate the world
        # can be done after initialisation as well
        # although the counts won't be stored
        self.legend = None
        if population:
            self.populate(population)
        
        # store the population variables
        model_reporters = {
                'entropy':'entropy',
                'temperature': 'temperature',
                'luminosity': 'luminosity',
            }
        for name in self.legend.keys():
            def pop_count(name):
                return lambda s: s.counts[name]
            model_reporters[name] = pop_count(name)
        
        self.datacollector = DataCollector(model_reporters)
        
    
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
        # creates a uid for each name type
        legend = {'empty': 0}
        
        # for each in the population
        for name, traits in population.items():
            
            # only add to legend on first go
            if self.legend is None:
                legend[name] = len(legend)
            elif name not in self.legend.keys():
                raise Exception("Adding %s to existing %s is bad practice, please make sure it's added at creation with initial 0.0" % (name, self.legend))
            
            # create the given % of them
            how_many = int(self.area*traits['initial'])
            
            # then place randomly on the grid
            positions = self.random_cells(how_many)
            
            for pos in positions:
                self.add_agent(pos, name, traits['albedo'])
        
        # for plotting heatmap
        self.legend = legend
    
    def add_agent(self, pos, name, albedo):
        """
        Adds an agent (Daisy)
        
        :pos: where they are to be added to (see random_cells())
        :name: type of daisy e.g. white or black
        :albedo: the albedo of the daisy
        """
        agent = Daisy(self.agents_ever, self, name, albedo)
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
    
    def calculate_temperature(self, albedo, current_temperature):
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
    
    def update_temperature(self, albedo, pos):
        """ Updates the albedo of a given cell"""
        x, y = pos
        t = self.calculate_temperature(albedo, self.temperatures[x][y])
        self.temperatures[x][y] = t
    
    def update_global_temperature(self):
        """
        Adjust for local temperatures within cells
        based on the temperatures of their local neighbourhood.

        Ported from the excellent
        https://github.com/mchozhang/DaisyWorld
        """
        
        # update agents cells
        for a in self.schedule.agents:
            self.update_temperature(a.albedo, a.pos)
        
        # update empty cells
        for pos in self.grid.empties:
            self.update_temperature(self.surface_albedo, pos)
        
        # update neighbourhoods
        # store in a new array to update all at once
        #new_temperatures = np.zeros((self.size, self.size))
        for cx in range(self.size):
            for cy in range(self.size):
                
                # average over neighbours
                absorbed = 0
                for nx, ny in self.grid.iter_neighborhood((cx,cy), moore=True):
                     absorbed += self.temperatures[nx][ny]
                absorbed /= 8
                
                # calculate ultimate temperature after diffusion
                self.temperatures[cx][cy] = (self.temperatures[cx][cy] + absorbed) / 2
        
        #self.temperatures = new_temperatures
    
    def step(self):
        """ Move the model one step on """
        # adjust the temperature
        self.update_global_temperature()
        
        # step all the agents
        # spawning or dieing the daisies
        self.schedule.step()
        
        # clear the cache
        for stat in self.STATS_TO_CLEAR:
            try:
                delattr(self, stat)
            except AttributeError:
                # ignore any clearing issues
                pass
        
        # store the data
        self.datacollector.collect(self)
        
        
    
    def simulate(self, ticks):
        for i in range(ticks):
            self.step()
    
    """
    Convience attributes
    """
    
    @cached_property
    def area(self):
        return int(self.size**2)
    
    @cached_property
    def df(self):
        return self.datacollector.get_model_vars_dataframe()
    
    """
    Key stats to store
    """
    
    STATS_TO_CLEAR = ['counts', 'entropy', 'temperature', 'df']
    
    @cached_property
    def counts(self):
        counts = dict([(k, 0) for k in self.legend.keys()])
        for a in self.schedule.agents:
            counts[a.name] += 1
        counts['empty'] = len(self.grid.empties)
        return counts
    
    @cached_property
    def entropy(self):
        return shannon_entropy(list(self.counts.values()), True)
    
    @cached_property
    def temperature(self):
        return self.temperatures.mean()
    
    """
    Plots
    """
    
    def grid_as_numpy(self, offset=0):
        data = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                cell = self.grid.get_cell_list_contents((x,y))
                if len(cell) > 1:
                    raise RuntimeError("Too many daisies in grid %s %s" % (x, y))
                elif len(cell) == 1:
                    data[x][y] = self.legend[cell[0].name]+offset
        return data
    
    def plot(self):
        fig, axes = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(15,10))
        self.plot_grid(axes[0][0])
        self.plot_entropy(axes[0][1])
        self.plot_temperature(axes[1][0])
        self.plot_global_temp(axes[1][1])
    
    def plot_grid(self, ax=None):
        offset = 1
        data = self.grid_as_numpy(offset)
        cmap = sns.color_palette("cubehelix_r", as_cmap=True)
        vmax = len(self.legend)+1
        sns.heatmap(data, cmap=cmap, ax=ax,
                xticklabels=False, yticklabels=False, vmin=0, vmax=vmax)
    
    def plot_temperature(self, ax=None):
        sns.heatmap(self.temperatures, ax=ax,
                   xticklabels=False, yticklabels=False)
    
    def plot_global_temp(self, ax=None):
        d = self.df.temperature
        sns.lineplot(data=d, ax=ax)
    
    def plot_entropy(self, ax=None):
        e = self.df.entropy
        g = sns.lineplot(data=e, ax=ax)
        g.set(ylim=(0, None))

