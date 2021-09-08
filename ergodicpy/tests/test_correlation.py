import numpy as np
import unittest

from ..correlation import ErgodicCorrelation


class TestCorrelation(unittest.TestCase):
    
    def test_thresholds(self):
        for o in [200,1000]:
            x = np.random.random(o)
            lt = [
                (np.ones(o), 0.1),
                (np.random.random(o), 0.05),
            ]
            for y, threshold in lt:
                self.assertTrue(ErgodicCorrelation(x, y).complexity < threshold, threshold)
    
            gt = [
                (x, 0.8),
                (-x, 0.82),
                (x**3, 0.6),
            ]
            for y, threshold in gt:
                self.assertTrue(ErgodicCorrelation(x, y).complexity > threshold, threshold)

    def test_specific(self):
        np.random.seed(268480)
        o = 1000
        x = np.random.random(o)
        nats = {'pearson': 1.0, 'spearman': 1.0, 'kendall': 1.0, 'complexity': 1.3938138437785665}
        bits = {'pearson': 1.0, 'spearman': 1.0, 'kendall': 1.0, 'complexity': 1.6741410414392122}

        for c, unit in ((nats, None),(bits, 'bits')):
            ec = ErgodicCorrelation(x,x, units=unit)
            for k, v in c.items():
                self.assertEqual(v, ec.correlations[k])



if __name__ == '__main__':
    unittest.main()