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
        steps=400,
        luminosity=lum)

if __name__ == '__main__':

    filename = 'datasets/daisy_standard.pickle'
    luminosities = np.linspace(0.48, 1.35, 101)

    with Pool() as pool:
        ess = {}
        count = 0
        for k, v in pool.imap_unordered(partial, luminosities):
            ess[k] = v
            count += 1
            print(count)

    with open(filename, 'wb') as fh:
        pickle.dump(ess, fh)
