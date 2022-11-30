import sys
sys.path.append(".")

import pickle
import numpy as np
from simulations.daisy_world.series import series
from multiprocessing import Pool
import time

def partial(lum):
    return lum, series(
        ensembles=20,
        steps=250,
        luminosity=lum,
        population={
            'white': {'albedo': 0.75, 'initial': 0.3},
            'black': {'albedo': 0.25, 'initial': 0.3},
        })

if __name__ == '__main__':

    filename = 'datasets/daisy_adjusted.pickle'
    luminosities = np.linspace(0.48, 1.7, 101)

    with Pool() as pool:
        ess = {}
        count = 0
        for k, v in pool.imap_unordered(partial, luminosities):
            ess[k] = v
            count += 1
            print(count)

    with open(filename, 'wb') as fh:
        pickle.dump(ess, fh)
