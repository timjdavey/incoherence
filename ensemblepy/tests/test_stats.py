import numpy as np
import unittest

from ..stats import *


class TestEntropy(unittest.TestCase):
    
    def test_measures(self):
        boost = 280

        case_inputs = [
            [[0,1],[0,1]],
            [[0,1],[1,0]],
            [[0,1],[1,0]],
            [[0.5,0.5],[0.5,0.5]], 
            [[0,1],[1,0],[1,0]],
            [[0,1],[0.5,0.5],[1,0]],
        ]

        case_outputs = [
            {'ensemble': 0.0, 'pooled': 0.0, 'divergence': 0.0, 'complexity': 0.0, 'tau2': 0.0, 'tau2p': 1.0, 'entropies': [0.0, 0.0], 'weights': [0.5, 0.5]},
            {'ensemble': 0.0, 'pooled': 1.0, 'divergence': 1.0, 'complexity': 1.0, 'tau2': 194.08121055678467, 'tau2p': 0.0, 'entropies': [0.0, 0.0], 'weights': [0.5, 0.5]},
            {'ensemble': 0.0, 'pooled': 1.0, 'divergence': 1.0, 'complexity': 1.0, 'tau2': 194.08121055678467, 'tau2p': 0.0, 'entropies': [0.0, 0.0], 'weights': [0.5, 0.5]},
            {'ensemble': 1.0, 'pooled': 1.0, 'divergence': 0.0, 'complexity': 0.0, 'tau2': 0.0, 'tau2p': 1.0, 'entropies': [1.0, 1.0], 'weights': [0.5, 0.5]},
            {'ensemble': 0.0, 'pooled': 0.9182958340544896, 'divergence': 0.9182958340544896, 'complexity': 0.9582775349837277, 'tau2': 178.22396712254758, 'tau2p': 0.0, 'entropies': [0.0, 0.0, 0.0], 'weights': [0.33333333, 0.33333333, 0.33333333]},
            {'ensemble': 0.3333333333333333, 'pooled': 1.0, 'divergence': 0.6666666666666666, 'complexity': 0.816496580927726, 'tau2': 129.3874737045231, 'tau2p': 0.0, 'entropies': [0.0, 1.0, 0.0], 'weights': [0.33333333, 0.33333333, 0.33333333]},
        ]

        for i, pmfs in enumerate(case_inputs):
            expected = case_outputs[i]
            mms = measures(pmfs, with_meta=True, units='bits', tau_boost=boost)

            for key in LEGEND.keys():
                if key in ['weights', 'entropies']:
                    np.testing.assert_array_almost_equal(mms[key], expected[key])
                else:
                    self.assertEqual(mms[key], expected[key])

        with self.assertRaises(ZeroDivisionError):
            measures([])


if __name__ == '__main__':
    unittest.main()