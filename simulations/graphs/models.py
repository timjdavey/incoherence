import networkx as nx
import numpy as np
import math


class Graph:
    """Simple wrapper for calculating some things about graphs"""

    def edges_counts(self):
        observations = []
        for node in self.graph:
            observations.append(len(self.graph[node]))
        return observations


class ERGraph(Graph):
    """ER Graph in particular"""

    def __init__(self, nodes, p, q=0):
        if q > 1.0 or q < 0:
            raise ValueError("q must be between 0 and 1.0")

        p_prime = (1 - q) * p + q * np.random.random()
        self.graph = nx.generators.random_graphs.erdos_renyi_graph(nodes, p_prime)
        self.nodes = nodes
        self.p = p
        self.q = q

    @property
    def var(self):
        return self.p * (1 - self.p)

    @property
    def connected(self):
        return nx.is_connected(self.graph)

    @property
    def connected_hist(self):
        if self.connected:
            return [0, 1]
        else:
            return [1, 0]
