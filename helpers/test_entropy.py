from .entropy import shannon_entropy, ErgodicEnsemble
import numpy as np
import unittest


class TestEntropy(unittest.TestCase):
    
    def test_shannon_entropy(self):
        # pre-normalised
        cases = [
            (0.0, [1]),
            (0.0, [1, 0, 0]),
            (1.0, [0.5, 0.5, 0]),
            (2.0, [0.25, 0.25, 0.25, 0.25]),
            # float error rounding
            (0.0, [1.0000001]), #7
        ]
        for entropy, observations in cases:
            np.testing.assert_almost_equal(shannon_entropy(observations), entropy)

        # normalise
        cases = [
            (0.0, [3]),
            (0.0, [5,5]),
            
            (1.0, [1,2]),
            (1.0, [5,5,0,0]),
            (1.0, [5,5,5,5,5,5,5,5,5,0,0,0,0,0,0,0,0,0]),
            
            (2.0, [1,2,3,4]),
            (2.321928094887362, [1,2,3,4,5]),

            # bin rounding
            (1.0, [0.5, 0.6, 1, 1.1]),
            (1.5219280948873621, [0, 0.9, 1, 1.1, 2]),
        ]
        for entropy, observations in cases:
            observations = np.array(observations)
            bins = np.arange(observations.max()+2)
            pdf, nbins = np.histogram(observations, bins=bins)
            np.testing.assert_almost_equal(shannon_entropy(pdf, True), entropy)

        # errors for not normalised
        with self.assertRaises(ValueError):
            shannon_entropy([])

        with self.assertRaises(ValueError):
            shannon_entropy([2,1])

        # rounding error catch
        with self.assertRaises(ValueError):
            shannon_entropy([1.000001]) #6

    def test_ergodic(self):
        np.random.seed(19680800)

        cases = [
            (2.1697928929727737, 2.1639188572981265,
                [np.random.power(5,1000)*10 for c in range(1000)]),
            (3.3219230427487183, 3.3153808928073305,
                [np.random.uniform(0,10,1000) for c in range(1000)]),
        ]
        
        for ergodic, ensemble, observations in cases:
            observations = np.array(observations)
            bins = np.arange(observations.max() + 2)
            ee = ErgodicEnsemble(observations, bins)
            self.assertEqual(ee.ergodic, ergodic)
            self.assertEqual(ee.ensemble, ensemble)
            np.testing.assert_almost_equal(ee.complexity, 0.0, 2)

        # test do they run
        #ee.plot()
        ee.stats()



if __name__ == '__main__':
    unittest.main()