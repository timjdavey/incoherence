import numpy as np
import unittest

from ..series import ErgodicSeries
from ..scan import ErgodicScan

class TestScan(unittest.TestCase):

    def test_fixed(self):
        np.random.seed(1283947)

        x_label = 'tester'
        steps = 2
        ensembles = 5

        measures = {'ensemble max': [0.30951977140256587, 0.48171163346143314], 'ensemble trend': [0.2087866837915709, 0.48171163346143314], 'ergodic max': [0.3250829733914482, 0.4862229646617923], 'ergodic trend': [0.25363894692169137, 0.4862229646617923], 'divergence max': [0.04485226313012047, 0.019641621777890195], 'divergence trend': [0.04485226313012047, 0.004511331200359181], 'complexity max': [0.3859157617880867, 0.1971480585825731], 'complexity trend': [0.3859157617880867, 0.07659594846773761]}

        # generate
        ys = []
        x = range(2)
        for i in x:
            y = [[np.random.power(5,20)*10 for _ in range(ensembles)] for _ in range(steps)]
            ys.append(ErgodicSeries(x=range(steps), y=y, x_label=x_label))

        scan = ErgodicScan(x=x, y=ys)
        for k in measures.keys():
            np.testing.assert_array_almost_equal(scan.measures[k], measures[k])
        scan.dataframe()


if __name__ == '__main__':
    unittest.main()