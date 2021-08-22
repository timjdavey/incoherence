import numpy as np
import unittest

from ..bins import binr, BinError


class TestBins(unittest.TestCase):

    def test_errors(self):
        # minimum too high
        with self.assertRaises(BinError):
            binr(observations=[[1,2,3],[1,2,3]], minimum=5)
        
        # maximum too low
        with self.assertRaises(BinError):
            binr(observations=[[1,2,3],[1,2,3]], maximum=2)

        # count too low
        with self.assertRaises(BinError):
            binr(observations=[[1,2,3],[1,2,3]], maximum=2, count=1)

    def test_cases(self):
        cases = [
            # defaults
            (binr(observations=[[0,1,3],[0,5]]),
                [0. , 2.5, 5. ]),
            (binr(observations=[[1,3],[5]]),
                [1., 3., 5.]),

            # minimums
            (binr(observations=[[1,3],[5]], minimum=0),
                [0,2.5,5]),

            # maximums
            (binr(observations=[[1,3],[5]], minimum=0, maximum=8),
                [0., 4., 8.]),

            # count
            (binr(observations=[[1,10],[10]], count=3),
                [ 1.,  4.,  7., 10.]),

            # decimals
            (binr(observations=[[0,1],[1]], count=5),
                np.linspace(0,1,6)),

            # ratios
            (binr(observations=[np.zeros(100),np.ones(100)]),
                np.linspace(0,1,11)),
            (binr(observations=[np.zeros(100),np.ones(100)], ratio=50),
                [0. , 0.5, 1. ]),

            # no obs
            # create int bins in basic case
            (binr(0, 5), [0,1,2,3,4,5]),
            (binr(2, 5), [2,3,4,5]),
            # create alt bin numbers
            (binr(2, 5, 8), [2. , 2.375, 2.75 , 3.125, 3.5  , 3.875, 4.25 , 4.625, 5.   ])
        ]
        for created, expected in cases:
            np.testing.assert_array_equal(created, expected)



if __name__ == '__main__':
    unittest.main()