import unittest
import numpy as np
from .models import CA1DEnsemble

class TestAutomata(unittest.TestCase):

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
            'Avg Cell Entropy': 0.9249603170398115,
            'Avg Stable Cell Entropy': 0.8713415946277461,
            'Last Cell Entropy': 0.9709505944546686,
            'Timesteps': steps,
            'Cells': cells,
        }
        df = ca.analysis
        for i in range(len(df)):
            for k, v in ensemble_data.items():
                np.testing.assert_almost_equal(df[k][i], v)

        # Then check appends ergodic data
        self.assertEqual(df['Ergodic'][0], False)
        self.assertEqual(df['Ergodic'][1], True)

