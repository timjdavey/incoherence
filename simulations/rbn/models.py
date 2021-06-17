import numpy as np
import networkx as nx

from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid


class Node(Agent):
    """ Simple Binary node with `state` as 0 or 1 """
    def __init__(self, uid, model, table, neighbours, initial):
        super().__init__(uid, model)
        self.table = table
        self.neighbours = neighbours
        self.state = initial
    
    def neighbour_states(self):
        states = []
        for n in self.neighbours:
            states.append(self.model.schedule.agents[n].state)
        return states
    
    def step(self):
        self.next_state = self.table.new_state(self.state, self.neighbour_states())
        
    def advance(self):
        if self.next_state not in [0,1]:
            raise Exception(
                "State is set to be %s which is not a 0 or 1"
                % self.next_state)
        self.state = self.next_state


class RandomBooleanNetwork(Model):
    """
    Generic model which can take any kind of network
    and unordered truth table evolve the network.
    """
    def __init__(self, network, table=None, states=None):
        
        # All fire at the same time
        self.schedule = SimultaneousActivation(self)
        # Set all the standard variables
        self.G = network
        self.grid = NetworkGrid(self.G)
        self.table = table        
        # Record the data
        self.datacollector = DataCollector(
            agent_reporters = {'state': 'state'},
            #model_reporters = {},
        )
        
        for i, node in enumerate(self.G.nodes()):
            # allow passing of initial states
            if states is None:
                initial = np.random.choice([0,1])
            else:
                initial = states[i]
            # create & place node
            a = Node(
                uid = i,
                model = self,
                table = self.table,
                neighbours = self.G[node],
                initial = initial
            )
            self.schedule.add(a)
            self.grid.place_agent(a, node)
    
    
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
    
    def run_model(self, distance):
        for i in range(distance):
            self.step()

    def to_numpy(self):
        df = self.datacollector.get_agent_vars_dataframe()
        return df.unstack().to_numpy()

    def plot(self, size=(10,10)):
        """ Plot as a cellular automata """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=size)
        ax.set_xlabel("Nodes")
        ax.set_ylabel("Steps")
        sns.heatmap(self.to_numpy(), cbar=False, ax=ax)

    def plot_network(self):
        nx.draw_circular(self.G)


        