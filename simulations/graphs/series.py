import ergodicpy as ep
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .models import ERGraph


def generate(nodes, ensembles, p_range, cp):
    y = []
    divs = []
    for p in p_range:
        if cp is not None: cp(p)
        obs = np.array([[ERGraph(nodes, p).connected] for i in range(ensembles)], dtype='uint8')
        y.append(obs)
        divs.append(ep.ErgodicEnsemble(obs, [0,1,2]).measures['divergence'])
    return y, divs


def plot(nodes, ensembles, x, ax, cp, log):
    y, divs = generate(nodes, ensembles, x, cp)

    # divergence
    h = sns.lineplot(x=x, y=divs, ax=ax, label='Divergence')
    # ln(n)/n
    ax.axvline(x=np.log(nodes)/nodes, color='r')
    # percentage
    sns.lineplot(x=x, y=np.hstack(np.sum(y, axis=1)/ensembles), ax=ax, label='Percentage', color='orange')
    
    h.set(ylim=(0,1))
    h.set_xlabel("p")
    if log:
        h.set_title("Log scale")
        h.set(xscale='log')
    else:
        h.set_title("Ergodic Divergence of ER graphs  (%s nodes, %s ensembles)" % (nodes, ensembles))


def series(nodes, ensembles=200, steps=100, log=False, cp=None):
    # two figs for log and not log
    fig, axes = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(15,5))

    # plot normal
    plot(nodes, ensembles, np.linspace(0.0, 1.0, steps+1), axes[0], cp, False)

    # plot log
    plot(nodes, ensembles, np.geomspace(steps**-1, 1.0, steps+1), axes[1], cp, True)

    cp("")
    return fig