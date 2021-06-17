import numpy as np

from .models import RandomBooleanNetwork
from .generators import gen_RBN_retry


class UnorderedTruthTable:
    """
    Truth table for Random Boolean Network
    where the nodes are considered unordered,
    so the truth table mapping collapses due to symmetry
    
    e.g. for a 3 input network rather than
    000 unique random boolean outcome
    001 random boolean outcome
    010 same as 001
    011 random boolean outcome
    100 same as 001
    101 same as 011
    110 same as 011
    111 unique random boolean outcome
    
    so the 2**n states compresses to n+1 states
    and therefore adjusts the relative probabilities

    Interesting tables to prepopulate it with are
    - Noisy output {0: 0, 1: 1, 2: 0, 3: 1}
    - Immediate death {0: 0, 1: 0, 2: 0, 3: 0}
    - Slow life, with sometimes death {0: 1, 1: 0, 2: 1, 3: 1}
    - Flip flop {0: 1, 1: 0, 2: 0, 3: 0}
    """
    def __init__(self, inputs, table=None):
        if table is None:
            table = {}
            for i in range(inputs+1):
                table[i] = np.random.choice([0,1])
        self.table = table
    
    def new_state(self, self_state, neighour_states):
        neighour_states.append(self_state)
        return self.table[np.sum(neighour_states)]


class ExactRBN(RandomBooleanNetwork):
    """
    Creates a network of exactly N nodes
    who all have exactly E edges each.
    """
    def __init__(self, nodes, edges, table=None):
        
        g = gen_RBN_retry(nodes, edges)
        
        if table is None:
            table = UnorderedTruthTable(edges+1)
        
        super().__init__(g, table)