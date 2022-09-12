import numpy as np
import unittest

from ..correlation import Correlation


class TestCorrelation(unittest.TestCase):
    
    def test_thresholds(self):
        for o in [200,1000]:
            x = np.random.random(o)

            # generated at random
            # so give or take are they close
            # these should be small complexities
            lt = [
                (np.ones(o), 0.1),
                (np.random.random(o), 0.05),
            ]
            for y, threshold in lt:
                c = Correlation(x, y).complexity
                self.assertTrue(c < threshold, (c, threshold))
            
            
            # these should broadly give large ones
            gt = [
                (x, 0.7),
                (-x, 0.72),
                (x**3, 0.6),
            ]
            for y, threshold in gt:
                c = Correlation(x, y).complexity
                self.assertTrue(c > threshold, (c, threshold))

    def test_specific(self):
        np.random.seed(268480)
        o = 1000
        x = np.random.random(o)
        nats = {'pearson': 1.0, 'spearman': 1.0, 'kendall': 1.0, 'complexity': 0.7517597472513868, 'is_complex': 1}
        bits = {'pearson': 1.0, 'spearman': 1.0, 'kendall': 1.0, 'complexity': 0.9029554784472786, 'is_complex': 1}

        for c, unit in ((nats, None),(bits, 2)):
            wec = Correlation(x,x, base=unit)
            
            # trying with or without weights, should give the same result, as same sized
            fec = Correlation(x,x, base=unit, weights=False)

            for k, v in c.items():
                self.assertEqual(v, wec.correlations[k])
                self.assertEqual(v, fec.correlations[k])



if __name__ == '__main__':
    unittest.main()