import numpy as np
import itertools

STEP_COUNT = 1000
VARIANCE_K = 50
STEP_RANGE = (-0.5, 1.5)

"""
Cache the max and min
values based on the default
values above

vmin, vmax = [], []
for d in range(1, 10):
    a, b = ep.minmax_variance(d)
    vmin.append(a)
    vmax.append(b)
    print(d)
print(vmin, vmax)
"""
MAX_VARIANCE = [0.00958671330400698, 0.006858584847388092, 0.005627527917502921, 0.00488695125774661, 0.0043785943541206065, 0.004001751754999471, 0.003707907251569037, 0.00347041218509296, 0.0032732479600455747]
MIN_VARIANCE = [0.0003759484743990121, 1.420760156238392e-06, 7.872466680609488e-09, 4.353257133122938e-10, 3.9376916622319765e-10, 9.065638644492943e-10, 2.247563768472738e-09, 2.3412008507151977e-10, 2.4635128780556455e-08]


def dimspace(lower, upper, count, dimensions, total_count=False):
    """
    Returns linspace for multiple dimensions
    
    :lower: lower value
    :upper: upper value
    :count: number of points
    :dimensions: dimensions of space
    :dimension_count: False, count means total points,
        if False means in per dimesion
    """
    if total_count: count = int(count**(1/dimensions)) 
    x = np.linspace(lower,upper,count)
    xs = [x for _ in range(dimensions)]
    return np.array(np.meshgrid(*xs)).T.reshape(-1, dimensions)

def spike(dimensions, count, val=1):
    """
    Returns an n-`dimensions` spike of `count` data at value `val`
    """
    return np.array([np.ones(count)*val for d in range(dimensions)])

def minmax_variance(dimensions=1, k=VARIANCE_K, var_range=STEP_RANGE, count=10000, steps=STEP_COUNT):
    """
    Calculates the minimum and maximum variance for a given k and var_range value.
    These should never realisically change from the defaults.
    
    :dimensions: number of dimensions of space
    :k: k value
    :var_range: var range
    :count: 10000 sample count
    """
    uniform = dimspace(0, 1, count, dimensions, True).T
    single = spike(dimensions, count)
    return tuple([densities(d, steps, k, var_range).var() for d in (uniform, single)])


def densities(data, steps=STEP_COUNT, k=VARIANCE_K, var_range=STEP_RANGE):
    """
    Returns the densities for a given `var_range`.
    Typically called within the context of density_variance().
    """
    data = np.array(data)

    # convert into single dimension array, so consistent for n-dimensions
    if data.ndim == 1:
        data = np.array([data,])

    if data.max() > 1 or data.min() < 0:
        raise ValueError("Please normalise `data` values to within [0,1]")

    dens = []
    for ref in dimspace(var_range[0], var_range[1], steps, data.ndim-1):
        dens.append(np.mean(np.exp(-k*np.linalg.norm(ref-data.T, axis=1))))
    return np.array(dens)


def density_variance(data, normalise=(0,1), steps=STEP_COUNT, bounded=True, k=VARIANCE_K, var_range=STEP_RANGE, vs=None):
    """
    Returns the density variance for a give set of `data`.

    :data: n-dimensional list of numbers normalised between [0,1].
    :steps: STEP_COUNT, the total number of discrete variance measures (higher is more accurate but computationally expensive).
    :bounded: True, ensures results are bounded within [0,1] handling any rounding errors within reason, erroring otherwise.
    :k: VARIANCE_K, the control factor to ensure appropriate variance dropoff at edges.
    :var_range: STEP_RANGE, the range over which variance is calculated.
    :vs: None, the upper and lower bounds, so can override if use different k or var_range values. If left blank, automatically calculates.
    """
    data = np.array(data)

    if len(data.shape) == 1:
        dims = 1  # handle if data was added as 1D array
    elif len(data.shape) == 2:
        dims = data.shape[0]
    else:
        raise ValueError("Please make sure data is in correct format of shape (dimensions, observations) not %s" % data.shape)


    if data.min() < normalise[0] or data.max() > normalise[1]:
        raise ValueError("Please make sure `data` %s is within `normalise` %s bounds" % ((data.min(), data.max()), normalise))
    else:
        data = (data-normalise[0])/(normalise[1]-normalise[0])

    dens = densities(data, steps, k, var_range)
    v = dens.var()

    if vs is None:
        if k == VARIANCE_K and var_range == STEP_RANGE and dims <= len(MAX_VARIANCE):
            v0, v1 = MIN_VARIANCE[dims-1], MAX_VARIANCE[dims-1]
        else:
            v0, v1 = minmax_variance(dims, k, var_range, steps=steps)
    else:
        v0, v1 = vs
    
    val = 1-(v-v0)/(v1-v0)

    if bounded:
        if val < 1.01 and val > -0.01:
            return max(min(val, 1.0),0.0)
        else:
            raise ValueError("Density variance %s is outside of bounded range [0,1]" % val)
    else:
        return val

