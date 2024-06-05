import numpy as np
import scipy as sp
from .entropy import ensemble_entropies, get_weights, pooled_entropy
from .densityvar import density_variance
from .divergences import js_divergence, radial_divergences


def _incoherence(p_entropy, entropies, maxent, weights=None):
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
        ndivs = js_divergence(p_entropy, entropies, weights, power=2)
        return np.sqrt(np.mean(ndivs) / (p_entropy * maxent))


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
    return (1 - density_variance(divergences, power=0.5, k=100)) ** 2, divergences


LEGEND = {
    "pooled": ("Entropy of pooled", "coral"),
    "incoherence": ("Incoherence", "blueviolet"),
    "cohesion": ("Cohesion", "teal"),
    "divergences": ("Radial divergences between each pair", "seagreen"),
    "entropies": ("Entropies of individual samples", "skyblue"),
    "weights": ("Weights of each sample", "red"),
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
        maxent = np.log2(len(data[0]))
    else:
        ents = np.array([density_variance(observations) for observations in data])
        weights = get_weights(data, weights, discrete=False)
        pooled = density_variance(np.concatenate(data).flatten())
        maxent = 1

    # is cumbersome, but allows for specific metrics to be called
    outs = {
        "entropies": ents,
        "weights": weights,
        "incoherence": _incoherence(pooled, ents, maxent, weights),
        "maxent": maxent,
        "pooled": pooled,
    }

    if metrics is not None and "cohesion" in metrics:
        outs["cohesion"], outs["divergences"] = _cohesion(data, discrete, ents)

    return outs
