from .continuous import Continuous
from .discrete import Discrete


def wrap(metric, discrete, *args, **kwargs):
    Model = Discrete if discrete else Continuous
    return Model(*args, metrics=(metric, ), **kwargs).measures[metric]

def incoherence(*args, **kwargs):
    return wrap('incoherence', *args, **kwargs)

def cohesion(*args, **kwargs):
    return wrap('cohesion', *args, **kwargs)