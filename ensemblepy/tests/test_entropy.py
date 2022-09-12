import numpy as np
import unittest

from ..entropy import *


class TestEntropy(unittest.TestCase):


    def test_weights(self):
        # no need to normalise as np does it
        np.testing.assert_array_equal(observation_weights([], [1,2,3]), [1,2,3])

        # actively turned off gives equal weights
        np.testing.assert_array_equal(observation_weights([[1,2],[1,2]], False), [0.5,0.5])

        # default weights are the counts
        cases = (
            ([[1,1],[1,1]],[0.5,0.5]), # normalises
            ([[5,2],[2,1]],[7/10,3/10]),
        )
        for in_weight, out_weight in cases:
            np.testing.assert_array_equal(observation_weights(in_weight), out_weight)


if __name__ == '__main__':
    unittest.main()