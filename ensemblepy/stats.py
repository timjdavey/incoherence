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
        divs = js_divergence(p_entropy, entropies, weights, power=2)
        return np.sqrt(divs / p_entropy)

def _cohesion(data, discrete=True):
    """
    Cohesion calculation from pmfs or observations,
    that is how `clumped` or `grouped` the ensembles are.
    
    :data: if discrete is True, in the form of histograms,
        if continuous just all the observations normalised between (0,1)
    :discrete: True, if data in histograms or False if continuous observations
    """
    divergences = radial_divergences(data, discrete)
    
    # if discrete need to normalise by bins (max entropy)
    # if continuous will already be normalised to (0,1)
    if discrete:
        divergences /= np.log(len(data[0]))

    return 1-density_variance(divergences)**2




LEGEND = {
    'ensemble': ('Mean entropy of individuals','orange'),
    'pooled': ('Entropy of pooled','firebrick'),
    'divergence': ('Divergence','forestgreen'),
    'incoherence': ('Incoherence','blueviolet'),
    'cohesion': ('Cohesion', 'teal'),
    'entropies': ('Entropies of individual ensembles','crest'),
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
    data = {}
    if metrics is None or 'ensemble' in metrics:
        data['ensemble'] = np.average(ents, weights=weights)
    
    if metrics is None or 'pooled' in metrics:
        data['pooled'] = pooled
    
    if metrics is None or 'divergence' in metrics:
        data['divergence'] =  js_divergence(pooled, ents, weights)
    
    if metrics is None or 'incoherence' in metrics:
        data['incoherence'] = _incoherence(pooled, ents, weights),
    
    if metrics is None or 'cohesion' in metrics:
        data['cohesion'] = _cohesion(data, discrete=discrete)
    
    if metrics is None or 'entropies' in metrics:
        data['entropies'] = ents
    
    if metrics is None or 'weights' in metrics:
        data['weights'] = weights

    print(data)
    return data

