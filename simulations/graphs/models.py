import networkx as nx


class Graph:
    """ Simple wrapper for calculating some things about graphs """
    def edges_counts(self):
        observations = []
        for node in self.graph:
            observations.append(len(self.graph[node]))
        return observations


class ERGraph(Graph):
    """ ER Graph in particular """
    def __init__(self, nodes, p):
        self.graph = nx.generators.random_graphs.erdos_renyi_graph(nodes, p)
        self.nodes = nodes
        self.p = p

    @property
    def var(self):
        return self.p*(1-self.p)
