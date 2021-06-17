import unittest

import numpy as np
import cellpylib as cpl

from .automata import CellularAutomata, gen_simple_state


class TestRBN(unittest.TestCase):
    
    def test_cellular_automata(self):
        import cellpylib as cpl

        timesteps = 100

        for cells in [10,100,200]:
            for rule in [30,10,200]:#range(1,255):

                # run using the better more efficient cellpylib framework
                cellular_automaton = cpl.init_simple(cells)
                cellular_automaton = cpl.evolve(cellular_automaton, timesteps=timesteps, 
                                        apply_rule=lambda n, c, t: cpl.nks_rule(n, rule))
        
                # run using the more generic oop network framework built on mesa
                abn_ca = CellularAutomata(cells, rule, gen_simple_state(cells))
                abn_ca.run_model(timesteps)
        
                # do they net the same results
                np.testing.assert_array_equal(abn_ca.to_numpy(), cellular_automaton)
