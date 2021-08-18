import numpy as np
import unittest

from .series import ErgodicSeries

class TestSeries(unittest.TestCase):

    def test_basic(self):
        ensembles = 10
        samples = 100
        steps = 20

        for ensembles in (2, 10):
            for samples in (10, 100):
                for steps in (1, 5):
                    y = [[np.random.power(5,samples)*10 for _ in range(ensembles)] for _ in range(steps)]
                    ees = ErgodicSeries(x=range(steps), y=y)
                    self.assertEqual(ees.entropies.shape, (steps, ensembles))
                    self.assertEqual(len(ees.bins), max(int(samples/10), 2)+1)
                    self.assertTrue(ees.complexity_max < 1)
                    self.assertTrue(ees.complexity_mean < 1)
                    self.assertTrue(ees.complexity_trend < 1)

if __name__ == '__main__':
    unittest.main()