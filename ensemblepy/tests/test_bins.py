import numpy as np
import unittest

from ..bins import *


class TestBins(unittest.TestCase):

    def test_binspace(self):
        np.testing.assert_array_almost_equal(binspace(0,1,2), [0.,0.5,1.])
        np.testing.assert_array_almost_equal(binspace(0,1,2,log=True), [0.,0.31622777,1.])
        np.testing.assert_array_almost_equal(binspace(0,1,5), [0.,0.2,0.4,0.6,0.8,1.])
        np.testing.assert_array_almost_equal(binspace(-0.5,1.5,3), [-0.5,0.16666667,0.83333333,1.5])

    def test_binint(self):
        np.testing.assert_array_equal(binint(0,1), [0,1])
        np.testing.assert_array_equal(binint(0,3), [0,1,2,3])
        np.testing.assert_array_equal(binint(0,10,2), [0,5,10])
        np.testing.assert_array_equal(binint(0,5,10), [0,1,2,3,4,5])
        np.testing.assert_array_equal(binint(0,5,8), [0,1,2,3,4,5])

    def test_ergs(self):
        obs = [[1,2],[3,4,5]]
        np.testing.assert_array_equal(pooled_obs(obs), [1,2,3,4,5])
        np.testing.assert_array_equal(binobs(obs), [1,2,3,4,5])

        series = [[[1,2],[3,4]],[[6,7],[8,9]]]
        np.testing.assert_array_equal(pooled_series(series), [[1,2],[6,7],[3,4],[8,9]])
        np.testing.assert_array_equal(binseries(series), [1, 3, 5, 7, 9])


if __name__ == '__main__':
    unittest.main()