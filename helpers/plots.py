import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def dual(observations, bins, labels=None,
        tidy_variable='variable', tidy_value='value', palette='flare', tidy_h='h',
        variable='ensemble'):
    
    """ Plots the histograms overlaid & the ergodic frequency plot """
    
    # tidy data
    tidy_ensembles = pd.melt(pd.DataFrame(observations, index=labels).T)
    tidy_ergo = pd.DataFrame({
            tidy_variable:tidy_h,tidy_value:np.concatenate(observations)})


    # Subplots
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(15,5))
    
    # Ensembles
    axes[0].set_title('Distributions by %s' % variable)
    # Colours - reverse colours
    legend = labels is not None and len(labels) < 12
    g = sns.histplot(ax=axes[0],
        bins=bins, element='step', fill=True, stat='density',
        data=tidy_ensembles, x=tidy_value, hue=tidy_variable,
        palette="%s_r" % palette, alpha=0.3, legend=legend)

    # Ergodic
    axes[1].set_title('Distribution of all observations (ergodic)')
    g = sns.histplot(ax=axes[1],
        bins=bins, element='step', stat='density',
        data=tidy_ergo, x=tidy_value, hue=tidy_variable,
        palette=palette, alpha=1.0, legend=False)



def ridge(observations, bins, tidy_variable='variable',
        tidy_value='value', palette = 'flare',
        title='Distributions of each ensemble, as a ridge plot for clarity'):
    """ Plots a ridge plot for a series of ensembles """

    # tidy data
    tidy_ensembles = pd.melt(pd.DataFrame(observations).transpose())

    # Ignore Tight layout warning
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=UserWarning)  

        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        # Initialize the FacetGrid object
        g = sns.FacetGrid(tidy_ensembles, row=tidy_variable,
            hue=tidy_variable, aspect=5, height=1.2, palette="%s_r" % palette)
        
        g.map(sns.histplot, tidy_value,
            element='step', bins=bins,
            fill=True, alpha=0.7)
    
        # Set the subplots to overlap
        g.fig.subplots_adjust(hspace=-0.5)
        # Remove axes details that don't play well with overlap
        g.fig.suptitle(title)
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=True, left=True)
