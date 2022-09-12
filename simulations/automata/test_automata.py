import unittest
import numpy as np
import ensemblepy as ep

from .models import r2e, CA1DEnsemble


class TestAutomata(unittest.TestCase):

    def test_r2e(self):
        cases = [
            np.ones(10),
            np.zeros(10),
            np.random.choice([0,1], size=200),
            np.random.choice([0,1], p=[0.1, 0.9], size=200),
        ]
        for c in cases:
            pmf, bins = np.histogram(c, bins=[0,1,2])
            # is my shannon_entropy calculator the same as the cpl calc
            # while testing the r2e which uses the cpl version
            np.testing.assert_almost_equal(ep.shannon_entropy(pmf, True, units='bits'), r2e(c))

    def test_creation(self):

        # basic creation routines work
        start_cases = {
            'simple': [[0,0,0,0,0,1,0,0,0,0]],
            'half': [[1,1,1,1,1,0,0,0,0,0]],
        }
        for k, v in start_cases.items():
            ca = CA1DEnsemble(30, 10, 2, init=k)
            for row in ca.raw:
                np.testing.assert_array_equal(row, v)

        # make sure the rest is the right structure
        cells = 10
        ensembles = 5
        for k in ['random', 'sparse']:
            ca = CA1DEnsemble(30, cells, ensembles, init=k)
            self.assertEqual(len(ca.raw), ensembles)
            for row in ca.raw:
                # number of runs so far (aka 1)
                self.assertEqual(len(row), 1)
                # number of cells
                self.assertEqual(len(row[0]), cells)

    def test_run_analysis(self):
        # cellpylib runs
        output = [
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 1, 0, 1, 1],
        [0, 1, 1, 1, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 1, 0, 0, 1, 1]]

        steps = 10
        cells = 10
        ca = CA1DEnsemble(30, cells, 1, init='half')
        ca.run(steps)
        np.testing.assert_array_equal(output, ca.raw[0])

        # analysis
        ca.analyse()

        # these values will be the same for ergodic as just 1 frame
        ensemble_data = {
            #'Avg Cell Entropy': 0.9249603170398115,
            'Avg Stable Cell Entropy': 0.8713415946277461,
            #'Last Cell Entropy': 0.9709505944546686,
            'Stable diag LR': 0.6959999571987501,
            'Stable diag RL': 0.5353554929439425,
        }
        df = ca.analysis
        # loop over the ensemble & ergodic
        for i in [0,1]:
            # then test data for ensembles
            for k, v in ensemble_data.items():
                np.testing.assert_almost_equal(df[k][i], v)
        
        # then check complexity numbers net to zero
        # as they're the same value
        i = 2
        for k, v in ensemble_data.items():
            np.testing.assert_almost_equal(df[k][i], 0)
        

if __name__ == '__main__':
    unittest.main()