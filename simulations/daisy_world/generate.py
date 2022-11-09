import sys
sys.path.append(".")

import pickle
import numpy as np
from simulations.daisy_world.series import series
from multiprocessing import Pool

def partial(lum):
    return series(
        ensembles=5,
        steps=10,
        luminosity=lum,
        population={
            'white': {'albedo': 0.75, 'initial': 0.3},
            'black': {'albedo': 0.25, 'initial': 0.3},
        })

if __name__ == '__main__':

    filename = 'simulations/daisy_world/scan.pickle'
    luminosities = np.linspace(0.48, 1.35, 2)

    with Pool() as pool:
        ess = []
        #for i in pool.imap_unordered(partial, luminosities):
        #    ess.append(i)
        for i in luminosities:
            ess.append(partial(i))

    with open(filename, 'wb') as fh:
        pickle.dump(ess, fh)
