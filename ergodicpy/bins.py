import numpy as np

class BinError(ValueError):
    """ When checking good bin structure """
    pass


def list_observations(observations):
    """ Given observations, returns the list and labels """
    if isinstance(observations, (list, np.ndarray)):
        return (observations, None)
    elif isinstance(observations, dict):
        return (list(observations.values()), observations.keys())
    else:
        raise TypeError(
            "`observations` is of type %s not list or dict" % type(observations))


def binr(minimum=None, maximum=None, count=None, observations=None, ratio=20):
    """
    Generates a set of bins given a list of observations.
    See test cases for examples.
    
    :count: _None_. the number of bins you want, leave None to use integers
    :minimum: _None_. the minimum value typically 0, leave None to use minimum found in observations
    :maximum: _None_. maximum is the max observed, adds +1 to catch upper bound
    :observations: _None_. list or dict of observations
    """

    if type(minimum) not in (float, int, type(None)):
        raise TypeError("minimum must be a float or int %s" % minimum)

    if observations is not None:
        observations, _ = list_observations(observations)
        observations = np.array(observations)
        all_observations = np.hstack(observations)
        
        amin = all_observations.min()
        amax = all_observations.max()

        if observations.shape[0] < 2:
            raise BinError("Need more than 1 ensemble in observations %s" % observations)
        
        if minimum is None:
            minimum = amin
        elif minimum > amin:
            raise BinError("minimum %s > observed min %s" % (minimum, amin))
        
        if maximum is None:
            maximum = amax
        elif maximum < amax:
            raise BinError("maximum %s < observed max %s" % (maximum, amax))

    # if count is None then have int bin widths, so add just one more int bin
    if count is None and observations is None:
        count = maximum-minimum

    # at least 20 per bin seems to reliable numbers as a default
    elif count is None and observations is not None:
        avg_per_ensemble = len(all_observations)/len(observations)
        count = max(int(avg_per_ensemble/ratio), 2)

    # obviously need at least 2 bin states to calculate entropy
    if count < 2:
        raise BinError("Count %s < 2, need at least 2 bins" % count)
    
    # we're dealing with edges so need to add the final edge with count
    return np.linspace(minimum, maximum, count+1)