import ergodicpy as ep
import numpy as np
from .models import MoneyModel


def series(agents=100, level=5, ensembles=20, steps=200, percent=None, threshold=None):
    """ Simple function to generate results for boltzmann wealth """
    x, y, models = [], [], []
    
    for _ in range(ensembles):
        models.append(MoneyModel(np.ones(agents), level, percent, threshold))
    
    # record initial positions
    x.append(0)
    y.append([m.observations() for m in models])
    
    # then step and record
    for i in range(1,steps+1):
        x.append(i)
        y.append([m.step() for m in models])
        
    # use log bins
    bins = ep.binr(minimum=0, series=y, log=True, ratio=2)
    ees = ep.ErgodicSeries(x=x, y=y, x_label='timesteps', bins=bins)
    ees.results()
    return ees