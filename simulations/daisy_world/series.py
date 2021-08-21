import numpy as np
import ergodicpy as ep

from .models import DaisyWorld, POP_DEFAULT


def scan(luminosities=np.linspace(0.4, 1.8, 50), *args, **kwargs):
    """ Does a scan of a range of a particular variable, in this case luminosities """
    
    # create an ErgodicSeries for each luminosity value
    ess = [series(luminosity=lum, *args, **kwargs) for lum in luminosities]

    # plot using normal scan
    sss = ep.ErgodicScan(x=luminosities, y=ess, trend=0.5)
    return sss, sss.plot()


def series(ensembles=20, steps=200, luminosity=1.0, population=POP_DEFAULT, mutate_p=0.0, mutate_a=0.05, cp=None):
    """ Creates an ErgodicSeries for a given set of inputs """

    # create worlds
    worlds = [DaisyWorld(population, luminosity=luminosity,
        mutate_p=mutate_p, mutate_a=mutate_a, store=False) for e in range(ensembles)]
    
    # initialise with initial distributions
    observations, x = [[w.observations() for w in worlds]], range(steps)

    # run for all worlds, for all time, always
    for d in x[1:]:
        if cp is not None: cp("luminosity: %s, steps left: %s" % (luminosity, steps-d))
        observations.append([w.step(True) for w in worlds])

    if cp is not None: cp("")
    observations = np.array(observations)
    # bins count species, so just the ints
    bins = ep.binr(0,observations.max()+1)
    es = ep.ErgodicSeries(x=x, y=observations, bins=bins, x_label='Timesteps')
    es.worlds = worlds
    return es