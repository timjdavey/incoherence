import numpy as np
import unittest

from ..series import ErgodicSeries

class TestSeries(unittest.TestCase):

    def test_random(self):
        for ensembles in (2, 10):
            for samples in (10, 100):
                for steps in (1, 5):
                    y = [[np.random.power(5,samples)*10 for _ in range(ensembles)] for _ in range(steps)]
                    ees = ErgodicSeries(x=range(steps), y=y)
                    self.assertEqual(ees.entropies.shape, (steps, ensembles))

                    # default ratio 10
                    self.assertEqual(len(ees.bins), max(int(samples/10), 2)+1)

                    # measures
                    for k, v in ees.measures.items():
                        self.assertEqual(len(v), steps)
                    
                    # just does it run
                    ees.dataframe()

                    # max / trends
                    for key, value in ees.max().items():
                        self.assertTrue(value >= ees.trend()[key])

    def test_fixed(self):
        np.random.seed(1283947)

        measures = {'ensemble': [0.30951977, 0.20878668],
                    'ergodic': [0.32508297, 0.25363895],
                    'divergence': [0.0155632 , 0.04485226],
                    'complexity': [0.19652311, 0.38591576]}

        mmax = {'ensemble': 0.30951977140256587, 'ergodic': 0.3250829733914482, 'divergence': 0.04485226313012047, 'complexity': 0.3859157617880867}
        mtrend = {'ensemble': 0.2087866837915709, 'ergodic': 0.25363894692169137, 'divergence': 0.04485226313012047, 'complexity': 0.3859157617880867}

        x_label = 'tester'
        steps = 2
        ensembles = 5

        # generate
        y = [[np.random.power(5,20)*10 for _ in range(ensembles)] for _ in range(steps)]
        ees = ErgodicSeries(x=range(steps), y=y, x_label=x_label)
        self.assertEqual(ees.x_label, x_label)

        # set measures
        for k, v in ees.measures.items():
            np.testing.assert_array_almost_equal(list(v), list(measures[k]))

        np.testing.assert_array_almost_equal(list(ees.max().values()), list(mmax.values()))
        np.testing.assert_array_almost_equal(list(ees.trend().values()), list(mtrend.values()))
        np.testing.assert_array_almost_equal(ees.peaks, [1])


if __name__ == '__main__':
    unittest.main()