import numpy as np
import scipy as sp
from itertools import combinations
from .entropy import ensemble_entropies, pooled_entropy
from .densityvar import density_variance


def js_divergence(p_entropy, entropies, weights, power=1):
    """Jenson Shannon Divergence"""
    divs = (p_entropy - entropies) ** power
    return np.average(divs, weights=weights)


def radial_divergences(data, discrete=True, entropies=None):
    """
    Returns the JS divergences for each pair of data

    :data: if discrete is True, in the form of histograms,
        if continuous just all the observations normalised between (0,1)
    :discrete: True, is the data histograms or continuous observations
    :entropies: list, saves calculating the individual entropies for data if already done
    """
    divergences = []
    if entropies is None:
        if discrete:
            entropies = ensemble_entropies(data)
        else:
            entropies = [density_variance(a) for a in data]

    indices = list(range(len(data)))
    for a, b in combinations(indices, 2):
        if discrete:
            p_entropy = pooled_entropy([data[a], data[b]])
        else:
            p_entropy = density_variance(np.concatenate([data[a], data[b]]))

        div = js_divergence(p_entropy, (entropies[a], entropies[b]), None)
        divergences.append(div)

    divergences = np.array(divergences)

    # if continuous will already be normalised to (0,1)
    if discrete:
        divergences /= np.log(2)

    # deal with float errors
    # divergences[divergences < 0] = 0
    # divergences[divergences > 1] = 1

    return divergences


def mean_divergence(data, func):
    """
    For a given histogram (discrete) data
    and a func, calculate the mean divergence
    between the pooled and each distribution.
    """
    pooled = np.mean(data, axis=0)
    return np.mean([func(h / np.sum(h), pooled / np.sum(pooled)) for h in data])


def kl_divergence(data, compare=None):
    return mean_divergence(data, sp.stats.entropy)


def hellinger(p, q):
    """
    https://en.wikipedia.org/wiki/Hellinger_distance
    https://gist.github.com/larsmans/3116927
    """
    return np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / 4
    # np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


def hellinger_distance(data):
    return mean_divergence(data, hellinger)


def wasserstein_distance(observations, discrete):
    pooled = np.concatenate(observations)
    if discrete:
        func = sp.stats.wasserstein_distance_nd
    else:
        func = sp.stats.wasserstein_distance
    return np.mean([func(pooled, obs) for obs in observations])


def total_variation(data):
    return mean_divergence(data, lambda p, q: 0.5 * np.sum(np.abs(p - q)))
