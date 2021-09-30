import numpy as np
import unittest

from ..ergodic import ErgodicEnsemble, ergodic_obs

class TestErgodic(unittest.TestCase):

    def test_basic(self):
        np.random.seed(19680800)

        ensembles = 1000
        samples = 1000

        cases = [
            ([np.random.power(5,samples)*10 for c in range(ensembles)], 
                [2.5313980896873165, 2.543260493286989, 0.011862403599672854, 0.017552312988442058, 615.5512151059751, 1.0]),
            ([np.random.uniform(0,10,samples) for c in range(ensembles)], 
                [0.6926254697950226, 0.6931469795819318, 0.0005215097869091837, 0.0010538755763709613, 2.2190861534815096, 1.0]),
        ]

        for observations, measures in cases:
            ee = ErgodicEnsemble(observations)
            # measures
            np.testing.assert_array_equal(list(ee.measures.values()), measures)
            self.assertEqual(len(ee.entropies), ensembles)

            # weights
            np.testing.assert_array_equal(ee.weights, np.ones(ensembles)*(1/ensembles))

            # obs
            self.assertTrue(ee.obs_min < 1.0)
            self.assertTrue(ee.obs_max > 9.99)
            self.assertEqual(ee.ensemble_count, ensembles)
            for v in ee.obs_counts.values():
                self.assertEqual(v, samples)


    def test_ergobs(self):
        np.testing.assert_array_equal(ergodic_obs([[1,2],[3,4,5]]), [1,2,3,4,5])

    def test_minimize(self):
        ensembles = 20

        for samples in [100,1000]:
            
            power_obs = [np.random.power(i, size=samples) for i in np.linspace(2, 3, ensembles)]
            uni_obs = [np.random.uniform(size=samples) for i in range(ensembles)]
            
            for obs in [uni_obs,power_obs]:
                
                ee = ErgodicEnsemble(obs)
                legacy = len(ee.bins)
        
                # do full scan & leave legacy bins
                scan = ee.bin_minimize(optimized=False, update=False, plot=True)
                self.assertEqual(legacy, len(ee.bins))
        
                # optimized & updates bins by default
                opti = ee.bin_minimize()
                self.assertEqual(opti[1][opti[0]][0], len(ee.bins)-1)
        
                # do they find the same complexity level
                # doesn't matter entirely about the exact bins
                opti_complexity = scan[1][scan[0]][1]
                scan_complexity = opti[1][opti[0]][1]
                np.testing.assert_almost_equal(opti_complexity, scan_complexity, 3)



if __name__ == '__main__':
    unittest.main()