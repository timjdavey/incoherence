import numpy as np
import unittest

from ..densityvar import *


class TestCorrelation(unittest.TestCase):
    
    def test_constants(self):
        for dimension in range(1,10):
            v0, v1 = minmax_variance(dimension)
            self.assertEqual(v0, MIN_VARIANCE[dimension])
            self.assertEqual(v1, MAX_VARIANCE[dimension])
            
            v0, v1 = minmax_variance(dimension, VARIANCE_K, STEP_RANGE)
            self.assertEqual(v0, MIN_VARIANCE[dimension])
            self.assertEqual(v1, MAX_VARIANCE[dimension])

    def test_densities(self):
        with self.assertRaises(ValueError):
            densities([2,4])

        self.assertEqual(len(densities([0.5,0.5])), STEP_COUNT)
        self.assertEqual(len(densities([0.5,0.5], steps=20)), 20)
        