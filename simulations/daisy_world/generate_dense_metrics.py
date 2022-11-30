import sys
sys.path.append(".")

import pickle
import numpy as np
from simulations.daisy_world.series import series
from multiprocessing import Pool
import time

def partial(lum):
    return lum, series(
        ensembles=200,
        steps=200,
        luminosity=lum,
        population={
            'white': {'albedo': 0.75, 'initial': 0.3},
            'black': {'albedo': 0.25, 'initial': 0.3},
        }).y[-1]

if __name__ == '__main__':

    filename = 'datasets/daisy_dense_metrics.pickle'
    luminosities = np.linspace(0.48, 1.35, 201)

    with Pool() as pool:
        ess = {}
        count = 0
        for k, v in pool.imap_unordered(partial, luminosities):
            ess[k] = v
            count += 1
            print(count)

    with open(filename, 'wb') as fh:
        pickle.dump(ess, fh)
