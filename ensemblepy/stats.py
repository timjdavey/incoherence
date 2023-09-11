import numpy as np
import scipy as sp
from .entropy import ensemble_entropies, get_weights, pooled_entropy
from .densityvar import density_variance
from .divergences import js_divergence, radial_divergences

def _incoherence(p_entropy, entropies, weights=None):
    """
    incoherence calculation from entropies

    :p_entropy: can use `pooled_entropy()` to calc
    :entropies: can use `entropies()` to calc
    :weights: _None_
    """
    if weights is None:
        weights = np.ones(len(entropies))
    if p_entropy == 0:
        return 0.0
    else:
        divs = js_divergence(p_entropy, np.array(entropies), weights, power=2)
        return np.sqrt(divs/p_entropy)

def _cohesion(data, discrete=True, entropies=None):
    """
    Cohesion calculation from pmfs or observations,
    that is how `clumped` or `grouped` the ensembles are.
    
    :data: if discrete is True, in the form of histograms,
        if continuous just all the observations normalised between (0,1)
    :discrete: True, if data in histograms or False if continuous observations
    :entropies: None, list of entropies saves recalculating here
    """
    divergences = radial_divergences(data, discrete, entropies)
    return (1-density_variance(divergences, power=0.5, k=100))**2, divergences




LEGEND = {
    'pooled': ('Entropy of pooled','coral'),
    'incoherence': ('Incoherence','blueviolet'),
    'cohesion': ('Cohesion', 'teal'),
    'divergences': ('Radial divergences between each pair', 'seagreen'),
    'entropies': ('Entropies of individual samples','skyblue'),
    'weights': ('Weights of each sample', 'red'),
}

def measures(data, weights=None, metrics=None, discrete=True, **kwargs):
    """
    Returns all metrics
    
    :data: the histograms by ensemble if discrete, otherwise the normalised observations if continuous
    :weights: _None_ calculates them automatically, _False_ ignores then, _list_ uses supplied
    :metrics: _None_, can specify which metrics you'd like returned in a list, otherwise will return all available
    :discrete: _True_ if the data is discrete, _False_ if continuous
    """
    if discrete:
        ents = ensemble_entropies(data, **kwargs)
        weights = get_weights(data, weights, discrete=True)
        pooled = pooled_entropy(data, weights, **kwargs)
    else:
        ents = np.array([density_variance(observations) for observations in data])
        weights = get_weights(data, weights, discrete=False)
        pooled = density_variance(np.concatenate(data).flatten())

    # is cumbersome, but allows for specific metrics to be called
    outs = {}
    
    if metrics is None or 'pooled' in metrics:
        outs['pooled'] = pooled
    
    if metrics is None or 'incoherence' in metrics:
        outs['incoherence'] = _incoherence(pooled, ents, weights)
    
    if metrics is None or 'cohesion' in metrics:
        outs['cohesion'], outs['divergences'] = _cohesion(data, discrete, ents)
    
    if metrics is None or 'entropies' in metrics:
        outs['entropies'] = ents
    
    if metrics is None or 'weights' in metrics:
        outs['weights'] = weights

    return outs

