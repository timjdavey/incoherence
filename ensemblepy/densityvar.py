import numpy as np
from scipy.stats import entropy
import itertools


DEFAULT_K = 10
DEFAULT_RANGE = (-0.5,1.5)
DEFAULT_STEPS = 500
DEFAULT_POWER = 1

CACHED_VS = {
    # standard: power = 1, but quick only steps=500
    (10, (-0.5, 1.5), 500, 1, 1): (0.007006952515415692, 0.040028626993849875),
    (10, (-0.5, 1.5), 500, 1, 2): (0.00040288367404277357, 0.003533380091049299),
    (10, (-0.5, 1.5), 500, 1, 3): (2.6236611203982995e-05, 0.0003378076786545754),

    # for cohesion: power = 0.5, but need higher drop-off so k=100
    (100, (-0.5, 1.5), 500, 0.5, 1): (0.0047683913076489465, 0.009711690148675985),
    (100, (-0.5, 1.5), 500, 0.5, 2): (0.00010905408911155288, 0.0001358670123656454),

    # for cellular automata more details
    (100, (-0.5, 1.5), 500, 1, 1): (9.696359378670741e-05, 0.005151090381558178),
    (100, (-0.5, 1.5), 500, 1, 2): (6.867003767957743e-08, 2.127189703421981e-05),
    (10, (-0.5, 1.5), 500, 0.5, 1): (0.02730861775895304, 0.06283397725602342),
    (10, (-0.5, 1.5), 500, 0.5, 2): (0.0064278794927177816, 0.012033246222506901),
    (1000, (-0.5, 1.5), 500, 0.5, 1): (0.0004985965422482039, 0.002063619720333923),
    (1000, (-0.5, 1.5), 500, 0.5, 2): (1.1802103705815318e-06, 7.079451229210704e-11),

    # for development: high resolution with steps=2000
    (10, (-0.5, 1.5), 2000, 1, 1): (0.007058290954639278, 0.040052248668986046),
    (10, (-0.5, 1.5), 2000, 1, 2): (0.0004060305619015262, 0.0036091902456439552),
    (10, (-0.5, 1.5), 2000, 1, 3): (2.6979262276843913e-05, 0.000354146896734922),
    (10, (-0.5, 1.5), 2000, 1, 4): (2.139069954703946e-06, 3.9605121737750026e-05)}


def generate_vs_cache(k=DEFAULT_K, var_range=DEFAULT_RANGE,
        steps=DEFAULT_STEPS, power=DEFAULT_POWER,
        dimension=1, vs=None, timings=False):
    """
    Generates the min & max variance values as a cache
    as dict with keys: tuple([k, (var_range lower, var_range upper), steps, power, dimension],)

    See CACHED_VS for example

    from ensemblepy.densityvar import generate_vs_cache
    vs = {}
    for d in [1,2]:
        for p in [0.5, 1]:
            for k in [10, 100]:
                vs = generate_vs_cache(k=k, dimension=d, power=p, vs=vs, timings=True)
    print(vs)
    """
    import time
    if vs is None: vs = {}
    params = (k, var_range, steps, power, dimension)
    tic = time.time()
    vs[params] = minmax_variance(*params)
    if timings: print(power, dimension, time.time()-tic)
    return vs


def dimspace(lower, upper, count, dimensions, limit_count=False):
    """
    Like linspace but for multiple dimensions
    """
    if limit_count: count = (int(count**(1/dimensions))+1)*dimensions
    x = np.linspace(lower, upper, count)
    # faster than itertools.product(*[x for _ in range(dimensions)])
    return np.concatenate(np.array(np.meshgrid(*[x for _ in range(dimensions)])).T).reshape(-1,dimensions)


def minmax_variance(k=DEFAULT_K, var_range=DEFAULT_RANGE,
        steps=DEFAULT_STEPS, power=DEFAULT_POWER,
        dimensions=1, count=5000):
    """
    Calculates the minimum and maximum variance for a given k and var_range value.
    These should never realisically change from the defaults.
    
    :dimensions: number of dimensions of space
    :k: k value
    :var_range: var range
    :count: 10000 sample count
    """
    # use random as much faster than dimspace(0,1,count,dimensions) and close enough with 10* count
    uniform = np.random.uniform(0,1,(count*10, dimensions))
    single = np.ones((count,dimensions))
    return tuple([_dv(power, dist, k, var_range, steps) for dist in (uniform, single)])


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
    So can expand to be parallelised with taichi in the future

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
    points = dimspace(*var_range, steps, dims, limit_steps)
    for ref in points:
        dens.append(np.mean(np.exp(-k*np.linalg.norm(ref-data, axis=1))))
    return np.array(dens), points


def _dv(power=DEFAULT_POWER, *args, **kwargs):
    """ Simplier version of densities to give raw density variance """
    return (densities(*args, **kwargs)[0]**power).var()


def density_variance(data, normalise=(0,1), bounded=True,
    k=DEFAULT_K, var_range=DEFAULT_RANGE, steps=DEFAULT_STEPS,
    power=DEFAULT_POWER, vs=CACHED_VS,
    limit_steps=True, auto_calculate=False):
    """
    Returns the density variance for a give set of `data` for a given `k` and or `var_range`

    This value is a continuous entropy estimator.

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

    v = _dv(power, data, k, var_range, steps, limit_steps)
    params = tuple([k, var_range, steps, power, dims])
    try:
        v0, v1 = vs[params]
    except KeyError:
        if auto_calculate:
            v0, v1 = minmax_variance(*params)
        else:
            raise ValueError("Params %s not found in `vs` keys. Please either set `auto_calculate`=True or use `generate_vs_cache()`" % params)

    dv = 1-(v-v0)/(v1-v0)

    if bounded:
        return min(max(dv, 0), 1)
    else:
        return dv





