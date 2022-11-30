import ensemblepy as ep
import numpy as np
from .models import MoneyModel

def generate(agents=100, level=5, ensembles=20, percent=None, threshold=None, initial=None):
    """ Generates the models """
    # set default initial
    if initial is None:
        initial = [np.ones(agents) for _ in range(ensembles)]
    
    return [MoneyModel(initial[i], level, percent, threshold) for i in range(ensembles)]


def series(agents=100, level=5, ensembles=20, steps=200, ratio=5, metrics=('incoherence', 'pooled', 'entropies'),
        percent=None, threshold=None, initial=None, cp=None, log=None, discrete=True, plot=True, plot_params=None):
    """
    Simple function to generate results for boltzmann wealth abm

    :agents: number of agents
    :level: the total wealth of the system which it's normalised to
    :ensembles: the number of ensembles to generate
    :steps: how many steps to run for
    :percent: the percentage wealth for the agents to transfer. None for 1 unit.
    :threshold: how complex you want the system to be. None for not at all.
    :initial: the 2D array of initial agent wealths (each row is an ensemble).
    :output: for printing progress e.g. print or nb.cp
    :log: if you need to use a log bin strategy, useful for large distribution spreads.

    :returns: an ep.Series objects
    """
    x, y = [], []
    models = generate(agents, level, ensembles, percent, threshold, initial)
    
    # record initial positions
    x.append(0)
    y.append([m.observations() for m in models])
    
    # then step and record
    for i in range(1,steps+1):
        if cp is not None: cp(steps-i)
        x.append(i)
        y.append([m.step() for m in models])
    
    x, y = np.array(x), np.array(y)
    if cp is not None: cp("calculate metrics")
    
    if discrete:
        # use log bins when using percent mode, as distribution becomes powerlaw
        if log is None: log = percent is not None
        bins = ep.binspace(y.min(), y.max(), agents/ratio, log)
        eps = [ep.Discrete(obs, bins=bins, metrics=metrics) for obs in y]
    else:
        eps = [ep.Continuous(obs, normalise=(y.min(), y.max()), metrics=metrics) for obs in y]

    if plot_params is None: plot_params = {}
    p = ep.plot_series(x, eps, **plot_params) if plot else None
    return x, eps, p