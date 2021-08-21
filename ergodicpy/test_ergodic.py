import numpy as np
import unittest

from .ergodic import ErgodicEnsemble

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
            self.assertEqual(ee.complexity, measures[3])
            self.assertEqual(len(ee.entropies), ensembles)
            ee.stats()

    def test_dicts(self):
        np.random.seed(19680800)

        first_samples = 200
        second_samples = 600

        obs = {
                'first': np.random.power(5,first_samples)*10,
                'second': np.random.power(5,second_samples)*10,
            }
        measures = [3.125397481164248, 3.198293599997641, 0.07289611883339298, 0.057043934109718836]
        ee = ErgodicEnsemble(obs)
        self.assertEqual(list(ee.labels), ['first', 'second'])
        np.testing.assert_array_equal(list(ee.measures.values()), measures)


if __name__ == '__main__':
    unittest.main()