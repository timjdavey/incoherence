import numpy as np
import unittest

from .entropy import shannon_entropy, measures, complexity_from_averages
from .bins import binr


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


    def test_complexity(self):
        cases = [
            # ensemble, ergodic, complexity
            (0, 0, 0.0,),
            (0.1, 0.1, 0.0),
            (10, 10, 0.0),
            (9, 10, 0.09999999999999998),
            (8, 10, 0.19999999999999996),
            (7, 10, 0.30000000000000004),
            (6, 10, 0.4),
            (5, 10, 0.5),
            (4, 10, 0.6),
            (3, 10, 0.7),
            (2, 10, 0.8),
            (1, 10, 0.9),
            (0, 20, 1.0),
            (0, 1, 1.0),
            (2, 1, 0.0),
        ]

        for c in cases:
            comp = complexity_from_averages(c[0], c[1])
            self.assertEqual(comp, c[2])

    def test_measures(self):
        cases = [
            ([[0,1],[0,1]], 0, 0, 0),
            ([[0,1],[1,0]], 0, 1, 1),
            ([[0,1],[1,0]], 0, 1, 1),
            ([[0.5,0.5],[0.5,0.5]], 1, 1, 0),
            ([[0,1],[1,0],[1,0]], 0, 0.9182958340544896, 1),
            ([[0,1],[0.5,0.5],[1,0]], 1.0/3, 1.0, 2.0/3),
        ]
        for c in cases:
            mms = measures(c[0], units='bits')
            self.assertEqual(mms[0], c[1]) # ensemble
            self.assertEqual(mms[1], c[2]) # ergodic
            self.assertEqual(mms[2], c[2]-c[1]) # divergence
            np.testing.assert_almost_equal(mms[3], c[3]) # complexity

        with self.assertRaises(ValueError):
            measures([])



if __name__ == '__main__':
    unittest.main()