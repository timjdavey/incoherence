import numpy as np
import unittest

from ..entropy import shannon_entropy, measures, complexity
from ..bins import binr


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

        # not normalised
        with self.assertRaises(ValueError):
            shannon_entropy([2,1])

        # rounding error catch
        with self.assertRaises(ValueError):
            shannon_entropy([1.000001]) #6 vs .7's


    def test_measures(self):
        cases = [
            ([[0,1],[0,1]], [0.0, 0.0, 0.0, 0.0, [0.0, 0.0]]),
            ([[0,1],[1,0]], [0.0, 1.0, 1.0, 1.0, [0.0, 0.0]]),
            ([[0,1],[1,0]], [0.0, 1.0, 1.0, 1.0, [0.0, 0.0]]),
            ([[0.5,0.5],[0.5,0.5]], [1.0, 1.0, 0.0, 0.0, [1.0, 1.0]]),
            ([[0,1],[1,0],[1,0]], [0.0, 0.9182958340544896, 0.9182958340544896, 0.9582775349837277, [0.0, 0.0, 0.0]]),
            ([[0,1],[0.5,0.5],[1,0]], [0.3333333333333333, 1.0, 0.6666666666666667, 0.816496580927726, [0.0, 1.0, 0.0]]),
        ]
        for pmfs, expected in cases:
            mms = list(measures(pmfs, with_entropies=True, units='bits').values())
            for i, e in enumerate(expected):
                self.assertEqual(mms[i], e)

        with self.assertRaises(ValueError):
            measures([])



if __name__ == '__main__':
    unittest.main()