import numpy as np
import unittest

from ..entropy import *


class TestEntropy(unittest.TestCase):
    
    def test_shannon_entropy(self):
        # pre-pmf'd
        cases = [
            (0.0, [1, 0, 0]),
            (1.0, [0.5, 0.5, 0]),
            (2.0, [0.25, 0.25, 0.25, 0.25]),
            # float error rounding
            (0.0, [0,1.0000001]), #7
            # normalise
            (1.0, [2, 2, 0]),
            (2.0, [2, 2, 2, 2]),
        ]
        for entropy, observations in cases:
            np.testing.assert_almost_equal(shannon_entropy(observations, normalise=True, units='bits'), entropy)

        # errors for empty
        with self.assertRaises(ValueError):
            shannon_entropy([])


    def test_measures(self):
        boost=2000

        cases = [
            ([[0,1],[0,1]], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, [0.5, 0.5], [0.0, 0.0]]),
            ([[0,1],[1,0]], [0.0, 1.0, 1.0, 1.0, 2000.0, 0.0, [0.5, 0.5], [0.0, 0.0]]),
            ([[0,1],[1,0]], [0.0, 1.0, 1.0, 1.0, 2000.0, 0.0, [0.5, 0.5], [0.0, 0.0]]),
            ([[0.5,0.5],[0.5,0.5]], [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, [0.5, 0.5], [1.0, 1.0]]),
            ([[0,1],[1,0],[1,0]], [0.0, 0.9182958340544896, 0.9182958340544896, 0.9582775349837277, 3673.1833362179577, 0.0, [1/3, 1/3, 1/3], [0.0, 0.0, 0.0]]),
            ([[0,1],[0.5,0.5],[1,0]], [0.3333333333333333, 1.0, 0.6666666666666666, 0.816496580927726, 2666.6666666666665, 0.0, [1/3, 1/3, 1/3], [0.0, 1.0, 0.0]]),
        ]
        for pmfs, expected in cases:
            mms = measures(pmfs, with_meta=True, units='bits', boost=boost)
            for i, key in enumerate(LEGEND.keys()):
                self.assertEqual(mms[key], expected[i])

        with self.assertRaises(ValueError):
            measures([])


    def test_weights(self):
        # test returns normalised weights by default if given them
        np.testing.assert_array_equal(observation_weights([], [1,2,3]), [1/6,2/6,3/6])

        # actively turned off gives equal weights
        np.testing.assert_array_equal(observation_weights([[1,2],[1,2]], False), [0.5,0.5])

        # default weights are the counts
        cases = (
            ([],[]),
            ([[1,1],[1,1]],[0.5,0.5]), # normalises
            ([[5,2],[2,1]],[7/10,3/10]),
        )
        for in_weight, out_weight in cases:
            np.testing.assert_array_equal(observation_weights(in_weight), out_weight)


if __name__ == '__main__':
    unittest.main()