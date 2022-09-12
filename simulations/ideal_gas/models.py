import numpy as np
import ensemblepy as ep

from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace


class Particle(Agent):
    """ Basic particle which bounces around """
    
    def __init__(self, uid, model, pos, box, temperature):
        super().__init__(uid, model)

        self.pos = np.array(pos)
        self.box = box # assign to particle for performance

        # normalise velocity * adjust for temperature
        v = np.random.random(2)*2-1
        self.velocity = (v / np.linalg.norm(v)) * temperature
        
    
    def step(self):
        # check if out of bounds & bounce if so
        expected_pos = self.pos + self.velocity
        for i, wh in enumerate(self.box):
            if expected_pos[i] > wh or expected_pos[i] < 0:
                self.velocity[i] = -self.velocity[i]
        
        # move to new position
        self.pos = self.pos + self.velocity
        self.model.space.move_agent(self, self.pos)
    
    @property
    def x(self): return self.pos[0]

    @property
    def y(self): return self.pos[1]


class IsolatedBox(Model):
    """ An isolated box with an ideal gas inside """

    def __init__(self, N=10, width=10, height=10, temperature=0.15, skew=1.0):
        self.height = height
        self.width = width
        self.num_particles = N

        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(width, height, True)
        self.bins = ep.binr(0, width) # int bins of width

        for i in range(N):
            # placement
            x = self.random.random()*width*skew
            y = self.random.random()*height*skew
            
            p = Particle(i, self, (x, y), (width,height), temperature)
            self.space.place_agent(p, (x, y))
            self.schedule.add(p)
        
        self.datacollector = DataCollector(
            model_reporters = {'entropy': 'entropy_x'},
            agent_reporters = {'x':'x','y':'y'}
        )
    
    @property
    def observations(self):
        return [a.x for a in self.schedule.agents]

    @property
    def entropy_x(self):
        "entropy of just the x axis as a simplified proxy"
        return ep.entropy_from_obs(self.observations, bins=self.bins)
    
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def simulate(self, distance):
        for i in range(distance):
            self.step()

    def plot(self, trace=True):
        import seaborn as sns
        from matplotlib import pyplot as plt

        agent_data = self.datacollector.get_agent_vars_dataframe()
        model_data = self.datacollector.get_model_vars_dataframe()
        
        if trace:
            fig, axes = plt.subplots(1, 2, sharex=False, figsize=(15,5))
        else:
            fig, axes = plt.subplots(1, 3, sharex=False, figsize=(15,5))
        
        # entropy
        sns.lineplot(ax=axes[0], data=model_data)
        axes[0].set_title("Entropy evolution over time")
        axes[0].set_ylim(bottom=0)
        axes[0].legend(facecolor='white')
        
        # trace
        if trace:
            sns.scatterplot(ax=axes[1], data=agent_data,
                x='x', y='y', hue='Step', size='Step',legend=None)
            axes[1].set_title("Trace of particles over time")
            axes[1].set_xlim(left=0, right=self.width)
            axes[1].set_ylim(bottom=0, top=self.height)
        else:
            # start
            sns.scatterplot(ax=axes[1],
                data=self.datacollector.get_agent_vars_dataframe().loc[0],
                x='x', y='y', hue='AgentID', palette='icefire', legend=None)
            axes[1].set_title('Starting positions of particles')
            axes[1].set_xlim(left=0, right=self.width)
            axes[1].set_ylim(bottom=0, top=self.height)
                                          
            # end
            sns.scatterplot(ax=axes[2],
                data=self.datacollector.get_agent_vars_dataframe().loc[self.schedule.steps-1],
                x='x', y='y', hue='AgentID', palette='icefire', legend=None)
            axes[2].set_title('Finishing positions of particles')
            axes[2].set_xlim(left=0, right=self.width)
            axes[2].set_ylim(bottom=0, top=self.height)


