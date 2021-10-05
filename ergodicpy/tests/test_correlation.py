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
        tau_boost = 280
        x = np.random.random(o)
        nats = {'pearson': 1.0, 'spearman': 1.0, 'kendall': 1.0, 'complexity': 0.9199466575025159, 'tau2p': 0.0}
        bits = {'pearson': 1.0, 'spearman': 1.0, 'kendall': 1.0, 'complexity': 1.1049685452144649, 'tau2p': 0.0}

        for c, unit in ((nats, None),(bits, 'bits')):
            wec = ErgodicCorrelation(x,x, units=unit, tau_boost=tau_boost)
            # in this case the same as equally random
            fec = ErgodicCorrelation(x,x, units=unit, tau_boost=tau_boost, weights=False) 
            for k, v in c.items():
                self.assertEqual(v, wec.correlations[k])
                self.assertEqual(v, fec.correlations[k])



if __name__ == '__main__':
    unittest.main()