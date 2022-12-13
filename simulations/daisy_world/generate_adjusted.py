import sys
sys.path.append(".")

import pickle
import numpy as np
import ensemblepy as ep
from simulations.daisy_world.series import series
from multiprocessing import Pool
import time

def partial(lum):
    worlds = series(
        ensembles=30,
        steps=400,
        luminosity=lum,
        population={'white': {'albedo': 0.75, 'initial': 0.3}, 'black': {'albedo': 0.25, 'initial': 0.3}})
    return lum, ([w.temperatures.flatten() for w in worlds], [w.observations() for w in worlds])

if __name__ == '__main__':

    filename = 'datasets/daisy_adjusted.pickle'
    luminosities = np.linspace(0.45, 1.45, 201)

    with Pool() as pool:
        ess = {}
        count = 0
        for k, v in pool.imap_unordered(partial, luminosities):
            ess[k] = v
            count += 1
            print(count)

    with open(filename, 'wb') as fh:
        pickle.dump(ess, fh)
