import numpy as np



class Parcel(object):
    """
    A simple object that gets passed through the nodes
    For instance a chip in a silicon factory
    """
    def __init__(self, created_at):
        self.created_at = created_at


class Node(object):
    """
    A node where a parcel needs to pass through
    For instance a toolchain in a silicon factory
    or a depot in a delivery chain
    """
    def __init__(self, average, std=3):
        self.average = average
        self.std = std
        self.parcel = None
        self.finished = 0

    def is_finished(self, time):
        return self.parcel is not None and time > self.finished

    def recieve(self, time, parcel):
        self.finished = time+np.random.normal(self.average, self.std)


class System(object):

    def __init__(self, setup, std=3):
        self.results = []
        self.layers = {}
        for layer, row in enumerate(setup):
            nodes = []
            for average in row:
                nodes.append(Node(average, std))
            self.layers[layer] = nodes

        self.index = list(layers.keys())
        self.index.reverse()
        self.time = time

    def empty_nodes(self, layer):
        empty = []
        for node in self.layers[layer]:
            if node.parcel is None:
                empty.append(node)
        return empty

    def last_layer(self):
        for node in self.layers[self.index[0]]:
            if node.is_finished(time):
                parcel = node.parcel
                self.results.append(self.time-parcel.created_at)
                node.parcel = None

    def previous_layers(self):
        for layer in self.index[1:]:
            for node in self.layers[layer]:
                if node.is_finished(time):
                    random_node_next_layer = np.random.choice(self.empty_nodes(layer+1))
                    random_node_next_layer.recieve(self.time, parcel)
                    node.parcel = None

    def first_layer(self):
        for node in self.layers[0]:
            if node.parcel is None:
                node.parcel = Parcel(self.time)


    def step(self):
        
        self.time += 1
        self.last_layer()
        self.previous_layers()
        self.first_layer()

        








