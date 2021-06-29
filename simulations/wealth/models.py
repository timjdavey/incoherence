import numpy as np
import seaborn as sns

from mesa import Agent, Model
from mesa.time import RandomActivation

from helpers.ergodic import ErgodicEnsemble


class MoneyAgent(Agent):
    """An agent with fixed initial wealth."""
    def __init__(self, unique_id, model, initial):
        super().__init__(unique_id, model)
        self.wealth = initial
    
    def step(self):
        if self.wealth < 1:
            return
        
        other_agent = self.random.choice(self.model.schedule.agents)
        transfer = 1
        other_agent.wealth += transfer
        self.wealth -= transfer

        
class MoneyModel(Model):
    """
    A model with some number of `agents`.
    
    """
    def __init__(self, wealths, level=1):
        self.schedule = RandomActivation(self)
        
        # normalize
        wealths = wealths * wealths.size / wealths.sum()
        # level up
        wealths *= level
        
        # Create agents
        for i, w in enumerate(wealths):
            a = MoneyAgent(i, self, w)
            self.schedule.add(a)

    def step(self):
        self.schedule.step()

