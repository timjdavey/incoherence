import numpy as np
import unittest

from ..series import Series
from ..scan import Scan

class TestScan(unittest.TestCase):

    def test_fixed(self):
        np.random.seed(1283947)

        x_label = 'tester'
        steps = 2
        ensembles = 5

        measures = {'ensemble max': [0.30951977140256587, 0.48171163346143314], 'ensemble trend': [0.2087866837915709, 0.48171163346143314], 'pooled max': [0.3250829733914482, 0.4862229646617923], 'pooled trend': [0.25363894692169137, 0.4862229646617923], 'divergence max': [0.04485226313012047, 0.019641621777890195], 'divergence trend': [0.04485226313012047, 0.004511331200359181], 'incoherence max': [0.3859157617880867, 0.1971480585825731], 'incoherence trend': [0.3859157617880867, 0.07659594846773761]}

        # generate
        ys = []
        x = range(2)
        for i in x:
            obs = [[np.random.power(5,20)*10 for _ in range(ensembles)] for _ in range(steps)]
            ys.append(Series(x=range(steps), observations=obs, x_label=x_label))

        scan = Scan(x=x, y=ys)
        for k in measures.keys():
            np.testing.assert_array_almost_equal(scan.measures[k], measures[k])
        scan.dataframe()


if __name__ == '__main__':
    unittest.main()