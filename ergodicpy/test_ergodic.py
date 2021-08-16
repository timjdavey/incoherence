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
                [4.578316190618324, 4.611125507245071, 0.032809316626747353, 0.007115251271126111]),
            ([np.random.uniform(0,10,samples) for c in range(ensembles)], 
                [5.608217983831986, 5.643831331149529, 0.0356133473175424, 0.006310136718825188]),
        ]

        for observations, measures in cases:
            ee = ErgodicEnsemble(observations)
            np.testing.assert_array_equal(ee.measures, measures)
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
        measures = (3.5828707715189196, 3.6305413029431524, 0.04767053142423272, 0.013130419804228022)
        ee = ErgodicEnsemble(obs)
        self.assertEqual(list(ee.labels), ['first', 'second'])
        np.testing.assert_array_equal(ee.measures, measures)
        self.assertEqual(ee.ensemble_count, 2)
        np.testing.assert_array_equal(ee.obs_counts, (200, 400.0, 600))


if __name__ == '__main__':
    unittest.main()