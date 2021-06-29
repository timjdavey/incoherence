import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def dual(observations, bins, labels=None,
        tidy_variable=None, tidy_value=None, palette='flare', tidy_h='h'):
    
    """ Plots the histograms overlaid & the ergodic frequency plot """
    sns.set_style('white')

    # Names & labels
    if tidy_value is None: tidy_value = 'value'
    if tidy_variable is None: tidy_variable = 'ensemble'
    legend = labels is not None and len(labels) < 12

    # tidy data
    tidy_ensembles = pd.melt(pd.DataFrame(observations, index=labels).T,
            var_name=tidy_variable, value_name=tidy_value)
    tidy_ergo = pd.DataFrame({
            tidy_variable:tidy_h,tidy_value:np.concatenate(observations)})

    # Subplots
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15,5))

    # Ensembles
    axes[0].set_title('Distributions by %s' % tidy_variable)
    # Colours - reverse colours
    g = sns.histplot(ax=axes[0],
        data=tidy_ensembles, bins=bins,
        x=tidy_value, hue=tidy_variable,
        element='step', fill=True, stat='probability',
        common_norm=False, multiple='dodge',
        palette=palette, alpha=0.2, legend=legend)
    g.legend

    # Ergodic
    axes[1].set_title('Distribution of all observations (ergodic)')
    g = sns.histplot(ax=axes[1],
        data=tidy_ergo, x=tidy_value, hue=tidy_variable,
        bins=bins, element='step', stat='probability',
        palette=palette, alpha=1.0, legend=False)



def ridge(observations, bins, labels, tidy_variable='ensemble',
        tidy_value='value', palette = 'flare',
        title='Distributions of each ensemble, as a ridge plot for clarity'):
    """ Plots a ridge plot for a series of ensembles """

    # tidy data
    tidy_ensembles = pd.melt(pd.DataFrame(observations, index=labels).T,
            var_name=tidy_variable, value_name=tidy_value)

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
            ax.text(-0.1, .2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)
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
