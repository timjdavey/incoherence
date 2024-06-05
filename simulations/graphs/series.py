import ensemblepy as ep
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .models import ERGraph


def percentage(graphs):
    return np.sum([g.connected for g in graphs]) / len(graphs)


def incoherence(graphs):
    return ep.Collection([graph.connected_hist for graph in graphs]).incoherence


def generate(nodes, ensembles, p_range, cp):
    percentages = []
    incs = []
    for p in p_range:
        if cp is not None:
            cp(p)
        graphs = [ERGraph(nodes, p) for i in range(ensembles)]
        incs.append(incoherence(graphs))
        percentages.append(percentage(graphs))
    return percentages, incs


def plot(nodes, ensembles, x, ax, cp, log):
    percentages, divs = generate(nodes, ensembles, x, cp)

    # divergence
    h = sns.lineplot(
        x=x, y=divs, ax=ax, color="purple", label="Incoherence", legend=not log
    )
    # percentage
    sns.lineplot(
        x=x,
        y=percentages,
        ax=ax,
        color="orange",
        label="Percentage connected",
        legend=not log,
    )
    # ln(n)/n
    ax.axvline(x=np.log(nodes) / nodes, color="r")
    ax.axhline(y=0.5, color="r", linestyle="--")

    h.set(ylim=(-0.1, 1.1))

    if log:
        h.set_xlabel("Probability of connection (log scale)")
        h.set(xscale="log")
    else:
        h.set_title("Standard ERGraph %s nodes, %s trials" % (nodes, ensembles))
        h.set_xlabel("Probability of connection")
    return h


def series(nodes, nodes2, ensembles=200, steps=100, log=False, cp=None):
    # two figs for log and not log
    fig, axes = plt.subplots(2, 2, sharex=False, sharey=True, figsize=(11, 9))

    for i, n in enumerate((nodes, nodes2)):
        # plot normal
        h = plot(n, ensembles, ep.binspace(0.0, 1.0, steps), axes[0][i], cp, False)

        # plot log
        g = plot(
            n,
            ensembles,
            ep.binspace(steps**-1, 1.0, steps, log=True),
            axes[1][i],
            cp,
            True,
        )

    cp("")
    return fig
