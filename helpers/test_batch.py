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
            'region' : 0.09515441290755167,
            'year': 0.031586279197447276,
            'Unnamed: 0': 0.023369887018318658
        }
        ees = ergodic_collection(self.df, self.dist_name, self.ensemble_names)

        for case, value in cases.items():
            self.assertEqual(ees[case].complexity, value)


    def test_complexity_stablise(self):
        cases = [
        ['5', 0.11900916480810919, 
            0.058989358498507394, 0.024815115492958117],
        ['10', 0.1057490471298752, 
            0.04001560879484689, 0.02413210504830776]]

        complexities = complexity_stablise(range(5, 15, 5), True,
                        self.df, self.dist_name, self.ensemble_names)
        np.testing.assert_array_equal(np.array(cases, dtype='object'), complexities)


if __name__ == '__main__':
    unittest.main()