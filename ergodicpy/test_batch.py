import numpy as np
import pandas as pd
import unittest

from .batch import complexity_stablise, ergodic_collection



class TestEntropy(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.read_csv("helpers/test_avocado.csv")
        self.dist_name = 'AveragePrice'
        self.ensemble_names = ['region', 'year', 'Unnamed: 0']

    def test_ergodic_collection(self):
        cases = {
            'region' : 0.0952617170210005,
            'year': 0.03182494438355454,
            'Unnamed: 0': 0.023984623197454757
        }
        ees = ergodic_collection(self.df, self.dist_name, self.ensemble_names)

        for case, value in cases.items():
            #print(case, ees[case].complexity)
            self.assertEqual(ees[case].complexity, value)


    def test_complexity_stablise(self):
        cases = [
            ['5',
                0.12411634073361455,
                0.05523847831352113,
                0.02871870019154088],
            ['10',
                0.10451702899946047,
                0.04175130081863654,
                0.024715022675939058]]

        complexities = complexity_stablise(
                            range(5, 15, 5), plot=False,
                            sigmoid=False,
                            df=self.df, dist_name=self.dist_name,
                            ensemble_names=self.ensemble_names)
        np.testing.assert_array_equal(
                                np.array(cases, dtype='object'), complexities)


if __name__ == '__main__':
    unittest.main()