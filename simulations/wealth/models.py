import numpy as np

from mesa import Agent, Model
from mesa.time import RandomActivation


class MoneyAgent(Agent):
    """An agent with fixed initial wealth."""
    def __init__(self, unique_id, model, initial):
        super().__init__(unique_id, model)
        self.wealth = initial
        self.gave_to = []
        self.received_from = []


    def step(self):
        # if don't have enough to give, do nothing
        if self.wealth > 0:

            # how much to transfer
            perc = self.model.percent
            if perc is None:
                transfer = 1
            else:
                transfer = int(self.wealth*perc)
    
            # choose an agent
            agents = self.model.schedule.agents
            ran = self.random.random()
            thres = self.model.threshold

            if self.gave_to and thres is not None and \
                ran > (len(self.gave_to)/len(agents))/thres:
                other_agent = self.random.choice(self.gave_to)
            else:
                other_agent = self.random.choice(self.model.schedule.agents)
    
            # then transfer
            other_agent.wealth += transfer
            other_agent.received_from.append(self)
            self.wealth -= transfer
            self.gave_to.append(other_agent)


        
class MoneyModel(Model):
    """
    A model with some number of `agents`.
    
    :wealths: a list of wealths, allowing flexibility in distribution
    :level: is normalised by this, allowing you to increase "temperature" of system
    :percent: {'int'|'givers'|'thankers'}
    :probability:
    """
    def __init__(self, wealths, level=1, percent=None, threshold=None):
        self.schedule = RandomActivation(self)
        self.percent = percent
        self.threshold = threshold
        
        wealths = np.array(wealths)
        # normalize
        wealths = wealths * wealths.size / wealths.sum()
        # level up
        wealths *= level
        
        # Create agents
        for i, w in enumerate(wealths):
            a = MoneyAgent(i, self, w)
            self.schedule.add(a)

    def observations(self):
        return [a.wealth for a in self.schedule.agents]

    def step(self):
        self.schedule.step()
        return self.observations()

