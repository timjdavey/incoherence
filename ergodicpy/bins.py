import numpy as np


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
        return np.unique([int(i) for i in binspace(minimum, maximum, count)])


