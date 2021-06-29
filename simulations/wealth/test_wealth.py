import unittest
import numpy as np
from .models import MoneyAgent, MoneyModel


class TestWealth(unittest.TestCase):

    def test_basic(self):
        count = 20
        level = 5
        cases = [
            np.ones(count),
            np.random.randint(0,5,count),
        ]
        for wealths in cases:

            m = MoneyModel(wealths, level)
            agent_wealths = np.array([a.wealth for a in m.schedule.agents])
            
            # number of agents is count
            self.assertEqual(len(agent_wealths), count)
    
            # total wealth in system is consistent with expected
            self.assertEqual(int(agent_wealths.sum()), int(level*count))
    
            m.step()
    
            new_agent_wealths = np.array([a.wealth for a in m.schedule.agents])
    
            # same total wealth in system
            self.assertEqual(int(new_agent_wealths.sum()), int(level*count))

            # wealth has moved about
            np.testing.assert_raises(AssertionError,
                np.testing.assert_array_equal, new_agent_wealths, agent_wealths)

