import numpy as np
from mesa import Agent, Model


class Daisy(Agent):
    def __init__(self, uid, model, name, albedo, lifespan=25):
        super().__init__(uid, model)
        self.name = name
        self.legend = model.legend[name]
        self.albedo = albedo
        self.age = 0
        self.lifespan = lifespan
    
    @property
    def temperature(self):
        """ Get the temperature from the world temperature store """
        return self.model.temperatures[self.pos[0]][self.pos[1]]
    
    def step(self):
        """ Move the agent along one step """
        self.age += 1
        
        # if too old, then die, otherwise propogate!
        if self.age >= self.lifespan:
            self.die()
        elif self.reproduce_if():
            self.reproduce()
    
    def die(self):
        """ Removes the agent. """
        self.model.schedule.remove(self)
        self.model.grid.remove_agent(self)
    
    def empty_neighbors(self):
        """
        Find the empty neighbours
        
        :returns: list of positions
        """
        # neighbors that are empty
        grid = self.model.grid
        neighbors = grid.get_neighborhood(self.pos, moore=True)
        empty_neighbors = []
        
        for n in neighbors:
            if grid.is_cell_empty(n):
                empty_neighbors.append(n)
        return empty_neighbors
    
    def reproduce(self):
        """ Reproduce itself into an empty neighbor cell """
        empty_neighbors = self.empty_neighbors()
        
        # must be cells to reproduce into
        if empty_neighbors:
            x, y = self.random.choice(empty_neighbors)

            # does it mutate?
            if np.random.random() < self.model.mutate_p:
                species = self.model.add_species(self.name, mutation=True)
                albedo = np.random.normal(self.albedo, self.model.mutate_a)
                self.model.add_agent((x, y), species, albedo)
            else:
                self.model.add_agent((x, y), self.name, self.albedo)
    
    def reproduce_if(self):
        """ Check if conditions are right to reproduce.
        It's wednesday, the recycling has been taken out, maybe
        wearing some business socks.
        
        :returns: bool
        """
        # parabola with a peak of 1, temperature in range of [5, 40]
        # will have positive threshold value and probability to propagate seeds
        threshold = \
            0.1457 * self.temperature - 0.0032 * self.temperature ** 2 - 0.6443

        # probability to obtain a seed from neighbor and grow a new daisy
        probability = np.random.uniform(0, 1)

        return probability < threshold