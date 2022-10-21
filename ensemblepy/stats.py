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


def _incohesion(data, discrete=True):
    """
    incohesion calculation from pmfs or observations
    
    :data: if discrete is True, in the form of histograms,
        if continuous just all the observations normalised between (0,1)
    :discrete: True, is the data histograms or continuous observations
    """
    divergences = radial_divergences(data, discrete)
    
    # if discrete need to normalise by bins (max entropy)
    # if continuous will already be normalised to (0,1)
    if discrete:
        divergences /= np.log(len(data[0]))

    return density_variance(divergences)



LEGEND = {
    'ensemble': ('Mean entropy of individuals','orange'),
    'pooled': ('Entropy of pooled','firebrick'),
    'divergence': ('Divergence','forestgreen'),
    'incoherence': ('Incoherence','blueviolet'),
    'incohesion': ('Incohesion', 'teal'),
    'entropies': ('Entropies of individual ensembles','crest'),
    'weights': ('Weights of each ensemble', 'red'),
}

def measures(data, weights=None, with_meta=False, discrete=True, **kwargs):
    """
    Returns all metrics

    :weights: _None_ calculates them automatically, _False_ ignores then, _list_ uses supplied
    :with_meta: _False_ adds additional calculated stats
    :discrete: _True_ if the data is discrete, _False_ if continuous
    """
    if discrete:
        ents = ensemble_entropies(data, **kwargs)
        weights = get_weights(data, weights, discrete=True)
        pooled = pooled_entropy(data, weights, **kwargs)
    else:
        ents = np.array([density_variance(observations) for observations in data])
        weights = get_weights(data, weights, discrete=False)
        pooled = density_variance(np.concatenate(data))

    metrics = {
        'ensemble': np.average(ents, weights=weights),
        'pooled': pooled,
        'divergence': js_divergence(pooled, ents, weights),
        'incoherence': _incoherence(pooled, ents, weights),
        'incohesion': _incohesion(data, discrete=discrete),
    }
    if with_meta:
        metrics['entropies'] = ents
        metrics['weights'] = weights
    return metrics

