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
                [3.1734469592390404, 3.1961886445549688, 0.02274168531592835, 0.007115251271126]),
            ([np.random.uniform(0,10,samples) for c in range(ensembles)], 
                [3.8873204834587227, 3.912005774742179, 0.024685291283456312, 0.006310136718825077]),
        ]

        for observations, measures in cases:
            ee = ErgodicEnsemble(observations)
            np.testing.assert_array_equal(ee.measures[:4], measures)
            self.assertEqual(ee.ensemble, measures[0])
            self.assertEqual(ee.ergodic, measures[1])
            self.assertEqual(ee.divergence, measures[2])
            self.assertEqual(ee.complexity, measures[3])

    def test_dicts(self):
        np.random.seed(19680800)

        first_samples = 200
        second_samples = 600

        obs = {
                'first': np.random.power(5,first_samples)*10,
                'second': np.random.power(5,second_samples)*10,
            }
        measures = (2.4834567735889754, 2.5164994680414767, 0.0330426944525013, 0.013130419804228133)
        ee = ErgodicEnsemble(obs)
        self.assertEqual(list(ee.labels), ['first', 'second'])
        np.testing.assert_array_equal(ee.measures[:4], measures)
        self.assertEqual(ee.ensemble_count, 2)
        np.testing.assert_array_equal(ee.obs_counts, (200, 400.0, 600))


if __name__ == '__main__':
    unittest.main()