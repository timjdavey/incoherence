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


if __name__ == '__main__':
    unittest.main()