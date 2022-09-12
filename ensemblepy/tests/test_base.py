import numpy as np
import unittest

from ..base import EnsembleComplexity

class TestBase(unittest.TestCase):

    def test_basic(self):
        np.random.seed(19680800)

        ensembles = 1000
        samples = 1000

        cases = [
            ([np.random.power(5,samples)*10 for c in range(ensembles)], 
                [2.568593658512911, 2.5809324563957965, 0.012338797882884089, 0.017490786678111787]),
            ([np.random.uniform(0,10,samples) for c in range(ensembles)], 
                [1.0975842797291155, 1.0986114430311615, 0.0010271633020456057, 0.001349917246302409]),
        ]

        for observations, measures in cases:
            ee = EnsembleComplexity(observations)
            # measures
            np.testing.assert_array_almost_equal(list(ee.measures.values()), measures)
            self.assertEqual(len(ee.entropies), ensembles)

            # weights
            np.testing.assert_array_equal(ee.weights, np.ones(ensembles)*(1/ensembles))

            # obs
            self.assertEqual(ee.ensemble_count, ensembles)
            for v in ee.obs_counts.values():
                self.assertEqual(v, samples)

    def test_minimize(self):
        ensembles = 20

        for samples in [100,1000]:
            
            power_obs = [np.random.power(i, size=samples) for i in np.linspace(2, 3, ensembles)]
            uni_obs = [np.random.uniform(size=samples) for i in range(ensembles)]
            
            for obs in [uni_obs,power_obs]:
                
                ee = EnsembleComplexity(obs, lazy=True)
        
                # do full scan & leave legacy bins
                scan = ee.stabilize(optimized=False, plot=True)
                opti = ee.stabilize()
                self.assertEqual(opti[1][opti[0]][0], len(ee.bins)-1)
        
                # do they find the same complexity level
                # doesn't matter entirely about the exact bins
                opti_complexity = scan[1][scan[0]][1]
                scan_complexity = opti[1][opti[0]][1]
                np.testing.assert_almost_equal(opti_complexity, scan_complexity, 3)


if __name__ == '__main__':
    unittest.main()