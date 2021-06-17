import cellpylib as cpl
import networkx as nx
import numpy as np

from .models import RandomBooleanNetwork


class WolframTruthTable:
    """
    Simple wrapper around the excellent cellpylib
    for NKS rules.
    """
    def __init__(self, rule):
        self.rule = rule

    def new_state(self, self_state, neighbour_states):
        return cpl.nks_rule([
            neighbour_states[0],
            self_state,
            neighbour_states[1]], self.rule)

def gen_random_state(cells):
    """ Generates a random noisy initial state set """
    return np.random.choice([0,1], size=cells)

def gen_simple_state(cells):
    """ Generates all zero cells except one central one """
    x = np.zeros(cells)
    x[int(cells/2)] = 1
    return x

def gen_half_state(cells):
    """ Generates half the cells are zero, the other half 1"""
    o2 = int(cells/2)
    return np.append(np.zeros(o2), np.ones(o2))


class CellularAutomata(RandomBooleanNetwork):
    """
    A more vanilla Cellular Automata version of the RBN
    using a circular graph.
    """
    def __init__(self, nodes, rule, states=None):
        
        g = nx.cycle_graph(nodes)
        table = WolframTruthTable(rule)
        super().__init__(g, table, states)