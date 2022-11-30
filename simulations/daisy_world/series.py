import numpy as np
import ensemblepy as ep

from .models import DaisyWorld, POP_DEFAULT


def series(ensembles=20, steps=200, luminosity=1.0, population=POP_DEFAULT,
    mutate_p=0.0, mutate_a=0.05, store=False, vary_age=False, cp=None):
    """ Creates an Series for a given set of inputs """

    # create worlds
    worlds = []
    for _ in range(ensembles):
        # allow varying populations
        pop = population() if callable(population) else population
        w = DaisyWorld(pop, luminosity=luminosity,
            mutate_p=mutate_p, mutate_a=mutate_a, store=store, vary_age=vary_age)
        w.simulate(steps)
        worlds.append(w)

    return worlds