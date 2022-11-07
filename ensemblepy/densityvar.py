import numpy as np
from scipy.stats import entropy
import itertools


DEFAULT_K = 20
DEFAULT_RANGE = (-0.1, 1.1)
DEFAULT_STEPS = 2000
DEFAULT_MODE = 'var'
DIMENSIONS = 10
MODES = {'var': lambda x: x.var(), 'entropy': lambda x: entropy(x.flatten()) }

# tuple([k, (var_range lower, var_range upper), steps],)
CACHED_VS = {(20, (-0.1, 1.1), 2000, 'var', 1): (0.0008574780498222127, 0.03523253847900814)}


def dimspace(lower, upper, count, dimensions, limit_count=False):
    """
    Like linspace but for multiple dimensions
    """
    if limit_count: count = int(count**(1/dimensions)*dimensions)
    x = np.linspace(lower, upper, count)
    # faster than itertools.product(*[x for _ in range(dimensions)])
    return np.concatenate(np.array(np.meshgrid(*[x for _ in range(dimensions)])).T).reshape(-1,dimensions)


def minmax_variance(k=DEFAULT_K, var_range=DEFAULT_RANGE,
        steps=DEFAULT_STEPS, mode=DEFAULT_MODE, dimensions=1, count=1000):
    """
    Calculates the minimum and maximum variance for a given k and var_range value.
    These should never realisically change from the defaults.
    
    :dimensions: number of dimensions of space
    :k: k value
    :var_range: var range
    :count: 10000 sample count
    """
    # use random as much faster than dimspace(0,1,count,dimensions) and close enough with 10* count
    uniform = np.random.uniform(0,1,(count*100, dimensions))
    single = np.ones((count,dimensions))
    return tuple([MODES[mode](densities(d, k, var_range, steps)) for d in (uniform, single)])


def _dimensions(data):
    """
    Cleans the data and returns dimension count
    """
    data = np.array(data)

    if len(data.shape) == 1:
        dims = 1  # handle if data was added as 1D array
        data = np.array([data,]).T
    elif len(data.shape) == 2:
        # if fewer observations than dimensions
        # then assume the input in wrong way round
        if data.shape[0] < data.shape[1]:
            data = data.T
        dims = data.shape[1]
    else:
        raise ValueError("Please make sure data is in correct format of shape (dimensions, observations) not %s" % data.shape)
    return data, dims


def densities(data, k=DEFAULT_K, var_range=DEFAULT_RANGE, steps=DEFAULT_STEPS, limit_steps=True):
    """
    Returns the densities for a given `var_range`.
    Typically called within the context of density_variance().
    
    :adjust_steps: reduces the total number of steps
    :parallel: True, run as imap_unordered
    """
    data, dims = _dimensions(data)

    if data.max() > 1 or data.min() < 0:
        raise ValueError("Please normalise `data` values to within [0,1] (%s,%s) " % (data.min(), data.max()))

    """
    In theory can do this as a vector
    However, in practice it's much slower
    Particularly the more steps you have
    Plus has the benefit of not being ram bound
    And instead can be done via parallelisation

    return np.mean(\
            np.exp(\
                -k* np.apply_along_axis(np.linalg.norm, 2,\
                    np.moveaxis( \
                        np.broadcast_to( \
                            dimspace(*var_range, steps, dims, False), \
                                (data.shape[0],steps**dims,dims)),
                    0,1) - data)\
            ),\
        axis=1)
    """
    dens = []
    for ref in dimspace(*var_range, steps, dims, limit_steps):
        dens.append(np.mean(np.exp(-k*np.linalg.norm(ref-data, axis=1))))
    return np.array(dens)



def density_variance(data, normalise=(0,1), bounded=True, k=DEFAULT_K,
    var_range=DEFAULT_RANGE, steps=DEFAULT_STEPS, mode=DEFAULT_MODE, vs=CACHED_VS, limit_steps=True):
    """
    Returns the density variance for a give set of `data` for a given `k` and or `var_range`

    :data: n-dimensional list of numbers.
    :bounded: True, ensures results are bounded within [0,1] handling any rounding errors within reason, erroring otherwise.
    :parameters: DEFAULT_PARAMS, so can override
    :vs: CACHED_VS, minimum and maximum density variances from minmax_variance, only use if calculating bulk, otherwise ignore and it will automatically calculate
    """
    data, dims = _dimensions(data)

    if normalise is None:
        normalise = (data.min(), data.max())
    elif data.min() < normalise[0] or data.max() > normalise[1]:
        raise ValueError("Please make sure `data` %s is within `normalise` %s" % ((data.min(), data.max()), normalise))

    data = (data-normalise[0])/(normalise[1]-normalise[0])

    dens = densities(data, k, var_range, steps, limit_steps)
    v = MODES[mode](dens)
    try:
        v0, v1 = vs[tuple([k, var_range, steps, mode, dims])]
    except KeyError:
        v0, v1 = minmax_variance(k, var_range, steps, mode, dims)

    if mode == 'var':
        dv = (v-v1)/(v0-v1)#1-(v-v0)/(v1-v0)
    elif mode == 'entropy':
        dv = (v-v1)/(v0-v1)


    if bounded:
        return min(max(dv, 0), 1)
    else:
        return dv





