import numpy as np
import ensemblepy as ep

from .models import DaisyWorld, POP_DEFAULT


def series(ensembles=20, steps=200, population=POP_DEFAULT, *args, **kwargs):
    """ Creates an Series for a given set of inputs """

    # create worlds
    worlds = []
    for _ in range(ensembles):
        # allow varying populations
        pop = population() if callable(population) else population
        w = DaisyWorld(pop, *args, **kwargs)
        w.simulate(steps)
        worlds.append(w)

    return worlds