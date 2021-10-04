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
                [2.5313980896873165, 2.543260493286989, 0.01749078667811227, 152.96380930961482, 1.0]),
            ([np.random.uniform(0,10,samples) for c in range(ensembles)], 
                [0.6926254697950226, 0.6931469795819318, 0.0005215097869091837, 0.001349917246302409, 0.9111382859323394, 1.0]),
        ]

        'ensemble': ensemble,
        'ergodic': ergodic,
        'divergence': diver,
        'complexity': comp,
        'tau2': tau2p[0],
        'tau2p': tau2p[1],

        for observations, measures in cases:
            ee = ErgodicEnsemble(observations)
            # measures
            #print(list(ee.measures.values()))
            np.testing.assert_array_equal(list(ee.measures.values()), measures)
            self.assertEqual(len(ee.entropies), ensembles)

            # weights
            np.testing.assert_array_equal(ee.weights, np.ones(ensembles)*(1/ensembles))

            # obs
            self.assertEqual(ee.ensemble_count, ensembles)
            for v in ee.obs_counts.values():
                self.assertEqual(v, samples)




if __name__ == '__main__':
    unittest.main()