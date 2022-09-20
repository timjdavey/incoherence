import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def dual(tidy_ensembles, tidy_ergo, bins, labels=None,
        tidy_variable='ensemble', tidy_value='value', palette='flare', tidy_h='h', title=None):
    
    """ Plots the histograms overlaid & the pooled frequency plot """

    sns.set_style('white')

    # Only show legend if have small size
    legend = labels is not None and len(labels) < 12

    # Subplots
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))

    # Ensembles
    if title is None:
        title = 'Distributions by %s' % tidy_variable
    axes[0].set_title(title)
    # Colours - reverse colours
    g = sns.histplot(ax=axes[0],
        data=tidy_ensembles, bins=bins,
        x=tidy_value, hue=tidy_variable,
        element='step', fill=True, stat='probability',
        common_norm=False, #multiple='dodge',
        palette=palette, alpha=0.2, legend=legend)

    # Pooled
    axes[1].set_title('Distribution of all observations (pooled)')
    g = sns.histplot(ax=axes[1],
        data=tidy_ergo, x=tidy_value, hue=tidy_variable,
        bins=bins, element='step', stat='probability',
        palette=palette, alpha=1.0, legend=False)
    return g



def ridge(tidy_ensembles, bins, labels=None,
        tidy_variable='ensemble', tidy_value='value', palette = 'flare',
        title='Distributions of each ensemble, as a ridge plot for clarity'):

    """ Plots a ridge plot for a series of ensembles """

    # Ignore Tight layout warning
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=UserWarning)  

        # make sure background colour is transparent
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        # Initialize the FacetGrid object
        g = sns.FacetGrid(tidy_ensembles, row=tidy_variable,
            hue=tidy_variable, aspect=5, height=0.8, palette=palette)
        
        g.map(sns.histplot, tidy_value,
            element='step', bins=bins, stat='probability',
            fill=True, alpha=0.6, common_norm=False)
        
        # labels
        def label(x, color, label):
            ax = plt.gca()
            ax.text(-0.1, .2, label, fontweight="bold",
                color=color, ha="left", va="center", transform=ax.transAxes)
        g.map(label, tidy_variable)

        # Set the subplots to overlap
        g.fig.subplots_adjust(hspace=-0.4)
        # Remove axes details that don't play well with overlap
        g.fig.suptitle(title)
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=True, left=True)

        # reset theme
        sns.set_style('white')
    return g

def scatter(tidy_ensembles, bins, tidy_variable='ensemble', tidy_value='value',
            palette='flare', jitter=0.5, alpha=0.7):
    """ Plots a stripplot recreating what a scatter plot of the original data might have looked like """
    sns.set_theme(style="white")#, rc={"figure.figsize":(15, 10)})
    g = sns.stripplot(data=tidy_ensembles, x=tidy_variable, y=tidy_value,
            palette=palette, jitter=jitter, alpha=alpha, size=2)
    return g



def combine_legends(ax1, ax2):
    """ When you twin axis, this helpfully combines the legends into a single legend """
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc=0)
    ax2.get_legend().remove()



