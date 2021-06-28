import numpy as np
import unittest

from .ergodic import ErgodicEnsemble


class TestEntropy(unittest.TestCase):

    def test_ergodic(self):
        np.random.seed(19680800)

        ensembles = 1000
        samples = 1000

        cases = [
            (2.1697928929727737, 2.1639188572981265,
                [np.random.power(5,samples)*10 for c in range(ensembles)]),
            (3.3219230427487183, 3.3153808928073305,
                [np.random.uniform(0,10,samples) for c in range(ensembles)]),
        ]
        
        for ergodic, ensemble, observations in cases:
            observations = np.array(observations)
            bins = np.arange(observations.max() + 2)
            ee = ErgodicEnsemble(observations, bins)

            # statistics
            self.assertEqual(ee.ergodic, ergodic)
            self.assertEqual(ee.ensemble, ensemble)
            np.testing.assert_almost_equal(ee.complexity, 0.0, 2)

            # getters
            self.assertEqual(len(ee.get_ergodic_observations()), ensembles*samples)

            # test do they run
            #ee.plot()
            #ee.stats()



if __name__ == '__main__':
    unittest.main()