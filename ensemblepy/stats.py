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

def _cohesion(data, discrete=True):
    """
    Cohesion calculation from pmfs or observations,
    that is how `clumped` or `grouped` the ensembles are.
    
    :data: if discrete is True, in the form of histograms,
        if continuous just all the observations normalised between (0,1)
    :discrete: True, if data in histograms or False if continuous observations
    """
    divergences = radial_divergences(data, discrete, normalise=True)
    return 1-density_variance(divergences, power=0.5)




LEGEND = {
    'pooled': ('Entropy of pooled','coral'),
    'incoherence': ('Incoherence','blueviolet'),
    'cohesion': ('Cohesion', 'teal'),
    'entropies': ('Entropies of individual ensembles','skyblue'),
    'weights': ('Weights of each ensemble', 'red'),
}

def measures(data, weights=None, metrics=None, discrete=True, **kwargs):
    """
    Returns all metrics

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
        outs['cohesion'] = _cohesion(data, discrete=discrete)
    
    if metrics is None or 'entropies' in metrics:
        outs['entropies'] = ents
    
    if metrics is None or 'weights' in metrics:
        outs['weights'] = weights

    return outs

