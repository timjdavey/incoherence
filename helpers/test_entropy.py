import numpy as np
import unittest

from .entropy import shannon_entropy, int_entropy, complexity



class TestEntropy(unittest.TestCase):
    
    def test_shannon_entropy(self):
        # pre-normalised
        cases = [
            (0.0, [1, 0, 0]),
            (1.0, [0.5, 0.5, 0]),
            (2.0, [0.25, 0.25, 0.25, 0.25]),
            # float error rounding
            (0.0, [0,1.0000001]), #7
        ]
        for entropy, observations in cases:
            np.testing.assert_almost_equal(shannon_entropy(observations), entropy)

        # normalise
        cases = [
            (0.0, [3]),
            (0.0, [5,5]),
            
            (1.0, [1,2]),
            (1.0, [5,5,0,0]),
            (1.0, [5,5,5,5,5,5,5,5,5,0,0,0,0,0,0,0,0,0]),
            
            (2.0, [1,2,3,4]),
            (2.321928094887362, [1,2,3,4,5]),

            # bin rounding
            (1.0, [0.5, 0.6, 1, 1.1]),
            (1.5219280948873621, [0, 0.9, 1, 1.1, 2]),
        ]
        for entropy, observations in cases:
            np.testing.assert_almost_equal(int_entropy(observations), entropy)

        # errors for not normalised
        with self.assertRaises(ValueError):
            shannon_entropy([])

        with self.assertRaises(ValueError):
            shannon_entropy([2,1])

        # rounding error catch
        with self.assertRaises(ValueError):
            shannon_entropy([1.000001]) #6

    def test_complexity(self):
        cases = [
            # ensemble, ergodic, complexity, sigmoid
            (0, 0, 0.0, 0.0),
            (0.1, 0.1, 0.0, 0.0),
            (10, 10, 0.0, 0.0),
            (9, 10, 0.09999999999999998, 0.02961714293482348),
            (8, 10, 0.19999999999999996, 0.14470087946071847),
            (7, 10, 0.30000000000000004, 0.38736483348305006),
            (6, 10, 0.4, 0.687573460372717),
            (5, 10, 0.5, 0.8860289610969977),
            (4, 10, 0.6, 0.9692676482913651),
            (3, 10, 0.7, 0.9973864276163549),
            (2, 10, 0.8, 1.0),
            (1, 10, 0.9, 1.0),
            (0, 20, 1.0, 1.0),
            (0, 1, 1.0, 1.0),
            (2, 1, 0.0, 0.0),
        ]

        for c in cases:
            comp = complexity(c[0], c[1])
            self.assertEqual(comp, c[2])




if __name__ == '__main__':
    unittest.main()