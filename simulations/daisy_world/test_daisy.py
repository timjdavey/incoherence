import unittest
from .models import DaisyWorld, POP_DEFAULT



class TestDaisy(unittest.TestCase):

    def test_netlogo(self):
        """
        Comparing results from the Northwestern NetLogo implementation
        http://www.netlogoweb.org/launch#http://www.netlogoweb.org/assets/modelslib/Sample%20Models/Biology/Daisyworld.nlogo
        
        All defaults set in model are the Netlogo defaults (e.g. width is 29 cells)
        """

        cases = [
            {'luminosity': 0.8,
            'temperature': 38.8,
            'black': 608.0,
            'white': 0.0},
            {'luminosity': 1.0,
            'temperature': 23.1,
            'black': 450.0,
            'white': 385.0},
            {'luminosity': 1.2,
            'temperature': 15.6,
            'black': 228.0,
            'white': 603.0},
            {'luminosity': 1.4,
            'temperature': 6.9,
            'black': 0.0,
            'white': 840.0},
        ]

        for case in cases:
            world = DaisyWorld(POP_DEFAULT, luminosity=case['luminosity'])
            world.simulate(500)
            means = world.df[-20:].mean()

            for key, expected in case.items():
                m = means[key]
                e = expected
                self.assertTrue(e*0.8 <= m <= e*1.2,
                    "%slum %s %s != %s" % (case['luminosity'], key, m, e))





if __name__ == '__main__':
    unittest.main()