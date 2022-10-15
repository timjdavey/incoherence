import numpy as np

STEP_COUNT = 1000
VARIANCE_K = 50
STEP_RANGE = (-0.5, 1.5)

# based on above
MAX_VARIANCE = 0.009599039158625379
MIN_VARIANCE = 0.000375434287422549


def minmax_variance(k=VARIANCE_K, var_range=STEP_RANGE, count=10000, steps=STEP_COUNT*10):
    """
    Calculates the minimum and maximum variance for a given k and var_range value.
    These should never realisically change from the defaults.

    :k: k value
    :var_range: var range
    :count: 10000 sample count
    """

    return densities(np.linspace(0,1,count), steps, k, var_range).var(), \
        densities(np.ones(count), steps, k, var_range).var()


def densities(data, steps=STEP_COUNT, k=VARIANCE_K, var_range=STEP_RANGE):
    """
    Returns the densities for a given `var_range`.
    Typically called within the context of density_variance().
    """
    data = np.array(data)
    if data.max() > 1 or data.min() < 0:
        raise ValueError("Please normalise `data` values to within [0,1]")

    return np.sum( \
        np.exp(-k*np.abs( \
                    np.full([len(data),steps], \
                        np.linspace(var_range[0],var_range[1],steps) \
                    ).T-data)), \
        axis=1)/len(data)


def density_variance(data, normalise=(0,1), steps=STEP_COUNT, bounded=True, k=VARIANCE_K, var_range=STEP_RANGE, vs=None):
    """
    Returns the density variance for a give set of `data`.

    :data: 1D list of numbers normalised between [0,1].
    :steps: STEP_COUNT, the total number of discrete variance measures (higher is more accurate but computationally expensive).
    :bounded: True, ensures results are bounded within [0,1] handling any rounding errors within reason, erroring otherwise.
    :k: VARIANCE_K, the control factor to ensure appropriate variance dropoff at edges.
    :var_range: STEP_RANGE, the range over which variance is calculated.
    :vs: None, the upper and lower bounds, so can override if use different k or var_range values. If left blank, automatically calculates.
    """
    data = np.array(data)
    if data.min() < normalise[0] or data.max() > normalise[1]:
        raise ValueError("Please make sure `data` %s is within `normalise` %s bounds" % ((data.min(), data.max()), normalise))
    else:
        data = (data-normalise[0])/(normalise[1]-normalise[0])

    dens = densities(data, steps, k, var_range)
    v = dens.var()

    if vs is None:
        if k == VARIANCE_K and var_range == STEP_RANGE:
            v0, v1 = MIN_VARIANCE, MAX_VARIANCE
        else:
            v0, v1 = minmax_variance(k, var_range)
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

