import numpy as np
import unittest

from ..ergodic import ErgodicEnsemble

class TestErgodic(unittest.TestCase):

    def test_basic(self):
        np.random.seed(19680800)

        ensembles = 1000
        samples = 1000

        cases = [
            ([np.random.power(5,samples)*10 for c in range(ensembles)], 
                [3.844483389200944, 3.889014934030624, 0.044531544829680136, 0.026058280651090625]),
            ([np.random.uniform(0,10,samples) for c in range(ensembles)], 
                [4.554780388320652, 4.605119956305231, 0.050339567984578615, 0.02367657443036134]),
        ]

        for observations, measures in cases:
            ee = ErgodicEnsemble(observations)
            np.testing.assert_array_equal(list(ee.measures.values()), measures)
            self.assertEqual(len(ee.entropies), ensembles)
            ee.stats()

    def test_stablize(self):

        for samples in [100,1000]:
            obs = [np.random.power(i, size=samples) for i in np.linspace(2, 3, 20)]
    
            ee = ep.ErgodicEnsemble(obs)
    
            legacy = len(ee.bins)
    
            # do full scan & leave legacy bins
            scan = ee.stablize(optimized=False, update=False)
            self.assertEqual(legacy, len(ee.bins))
    
            # optimized & updates bins by default
            opti = ee.stablize() 
            self.assertEqual(opti[1][opti[0]][0], len(ee.bins)-1)
    
            opti_complexity = scan[1][scan[0]][1]
            scan_complexity = opti[1][opti[0]][1]
    
            # do they find the same complexity level
            # doesn't matter entirely about the exact bins
            np.testing.assert_almost_equal(opti_complexity, scan_complexity, 3)



if __name__ == '__main__':
    unittest.main()