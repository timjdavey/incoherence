import numpy as np

class BinError(ValueError):
    """ When checking good bin structure """
    pass


def observations_from_series(observations=None, series=None):
    """ Takes observations and series,
    Returns series as a set of observations.
    """
    if observations is not None and series is not None:
        raise InputError("Must only specify series or observations")
    elif series is not None:
        return np.stack(np.hstack(np.stack(series, axis=2)),axis=1), len(series)
    elif observations is not None:
        return observations, len(observations)
    else:
        return None, None


def binm(observations=None, series=None, ratio=1):
    """
    Bin magic.

    Creates bins based on the ergodic observations.
    """
    observations, ensembles = observations_from_series(observations, series)
    srtd = np.sort(np.hstack(observations))
    count = int(np.log2(len(srtd)/(ensembles))/ratio)
    indxs = np.linspace(0, len(srtd)-1, max(count,3))
    return [srtd[int(i)] for i in indxs]


def binc(series, count, log=False):
    """
    Wrapper to build bind based on a series of observations

    Same inputs as `bino` but with series inputed rather than obs
    """
    # horrible mess of a transformation
    # but simulates as if all steps in the series are just ensembles
    obs = np.stack(np.hstack(np.stack(series, axis=2)),axis=1)
    return bino(obs, count, log)


def bino(observations, count, log=False):
    """
    Wrapper to build bins based on observations

    :observations: 2d list of obs
    :count: the number of bins
    :log: _False_, if you want the bins in log format
    """
    # handle ragged stacks
    if len(set([len(o) for o in observations])) != 1:
        observations = np.array(observations, dtype=object)
    else:
        observations = np.array(observations)
    all_observations = np.hstack(observations)
    
    # find min & max
    minimum = all_observations.min()
    maximum = all_observations.max()

    # adjust count as dealing with edges
    count = int(count)+1

    if log:
        # doesn't accept 0 as an input, so fudge first bin
        if minimum == 0:
            arr = np.geomspace(0.1, maximum, count)
            arr[0] -= 0.1
            return arr
        else:
            return np.geomspace(minimum, maximum, count)
    else:
        return np.linspace(minimum, maximum, count)


def bini(minimum, maximum):
    """
    Creates integer bins
    """
    return np.arange(minimum, maximum+1)


def binr(minimum=None, maximum=None, count=None, observations=None, series=None,
                boost=2, log=False, min_default=3, max_default=20):
    """
    Bin regular.

    Generates a set of bins given a list of observations.
    See test cases for examples.
    
    :count: _None_. the number of bins you want, leave None to use integers
    :minimum: _None_. the minimum value typically 0, leave None to use minimum found in observations
    :maximum: _None_. maximum is the max observed, adds +1 to catch upper bound
    :observations: _None_. list or dict of observations
    """

    #
    # handle series inputs
    #
    observations, _ = observations_from_series(observations, series)
    
    #
    # make sure don't pass observations to minimum
    #
    try:
        iter(minimum)
    except TypeError:
        pass
    else:
        raise TypeError("minimum must be a float or int %s %s" % (minimum, type(minimum)))

    #
    # generating min, max
    #
    if observations is not None:

        
        

        # min & max

        if minimum is None:
            minimum = amin
        elif minimum > amin:
            raise BinError("minimum %s > observed min %s" % (minimum, amin))
        
        if maximum is None:
            maximum = amax
        elif maximum < amax:
            raise BinError("maximum %s < observed max %s" % (maximum, amax))


        # default count based on observations
        if count is None:
            count = np.round(np.log(avg_per_ensemble*boost))
            count = max(min_default, count) # standard minimum 3 bins
            count = min(max_default, count) # standard maximum 20 bins

    else:
        # default is ints
        if count is None:
            count = int(maximum-minimum)

    # + obviously need at least 2 bin states to calculate entropy
    if count < 2:
        raise BinError("Count %s < 2, need at least 2 bins" % count)
    else:
        # we're dealing with edges so need to add +1 to the final edge with count
        count = int(count) + 1
    
    #
    # for handling power law distributions better
    #
    if log:
        # doesn't accept 0 as an input, so fudge first bin
        if minimum == 0:
            arr = np.geomspace(0.1, maximum, count)
            arr[0] -= 0.1
            return arr
        else:
            return np.geomspace(minimum, maximum, count)
    else:
        return np.linspace(minimum, maximum, count)


