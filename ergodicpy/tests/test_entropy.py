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