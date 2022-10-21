import numpy as np
import ensemblepy as ep

from .models import DaisyWorld, POP_DEFAULT


def scan(luminosities=np.linspace(0.4, 1.8, 50), *args, **kwargs):
    """ Does a scan of a range of a particular variable, in this case luminosities """
    
    # create an Series for each luminosity value
    ess = [series(luminosity=lum, *args, **kwargs) for lum in luminosities]

    # plot using normal scan
    sss = ep.Scan(x=luminosities, y=ess, trend=0.5, x_label='Luminosity')
    return sss


def series(ensembles=20, steps=200, luminosity=1.0, population=POP_DEFAULT, mutate_p=0.0, mutate_a=0.05, store=False, vary_age=False, cp=None):
    """ Creates an Series for a given set of inputs """

    # create worlds
    worlds = []
    for _ in range(ensembles):
        # allow varying populations
        pop = population() if callable(population) else population
        worlds.append(DaisyWorld(pop, luminosity=luminosity,
            mutate_p=mutate_p, mutate_a=mutate_a, store=False, vary_age=vary_age))
    
    # initialise with initial distributions
    observations, x = [[w.observations() for w in worlds]], range(steps)

    # run for all worlds, for all time, always
    for d in x[1:]:
        if cp is not None: cp("luminosity: %s, steps left: %s" % (luminosity, steps-d))
        observations.append([w.step(True) for w in worlds])

    if cp is not None: cp("")
    observations = np.array(observations)

    # bins count species, so just the ints
    # this works are observations are integer labeled
    bins = ep.binint(0,observations.max()+1)
    es = ep.Series(x=x, observations=observations, bins=bins, x_label='Timesteps', models=worlds)
    return es