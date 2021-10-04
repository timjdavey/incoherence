import numpy as np


def ergodic_obs(observations):
    """
    Given a set of observations, returns the ergodic version.
    Handles ragged stacks, when the observations count isn't consistent across ensemble.
    """
    if len(set([len(o) for o in observations])) != 1:
        observations = np.array(observations, dtype=object)
    else:
        observations = np.array(observations)
    return np.hstack(observations)


def ergodic_series(series):
    """
    Given a series, returns the ergodic collection of all data
    i.e. if all the data was taken from a single model step
    """
    obs = np.stack(np.hstack(np.stack(series, axis=2)),axis=1)
    return obs


def binobs(observations, ratio=5):
    """
    Creates bins based on the observed min & max
    With a count which is average_obs_per_ensemble/`ratio=5`
    """
    ergobs = ergodic_obs(observations)
    count = max(3, (len(ergobs)/len(observations))/ratio)
    return binspace(ergobs.min(), ergobs.max(), count)


def binseries(series, ratio=5):
    """
    Same as bin_obs
    """
    return binobs(ergodic_series(series), ratio)


def binspace(minimum, maximum, count, log=False):
    """
    Like np.linspace but adjusts for bin count (as in +1)
    And allows you switch between log and linear via param.
    """
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


def binint(minimum, maximum, count=None, log=False):
    """
    Creates integer bins
    """
    if count is None:
        return np.arange(minimum, maximum+1)
    else:
        return np.unique([int(i) for i in binspace(minimum, maximum, count, log)])


