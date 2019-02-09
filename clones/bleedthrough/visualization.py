from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from .resampling import resample_uniformly
from .background import BackgroundExtraction
from ..visualization.settings import *


class CorrectionVisualization:
    """ Methods for visualizing correction procedure. """

    def show_fit(self, mode='box', bin_size=0.05, figsize=(3, 2)):
        """ Plot fit to background pixels using sns.boxplot . """

        # intantiate figure
        fig, ax = plt.subplots(figsize=figsize)

        # compile dataframe
        bg_xy = np.vstack((self.x, self.y)).T
        df = pd.DataFrame.from_records(bg_xy, columns=['x', 'y'])

        # add data to plot
        df['bin'] = (df.x // bin_size)
        if mode == 'strip':
            sns.stripplot(x='bin', y='y', data=df, ax=ax, size=1, color='k')
        elif mode == 'box':
            sns.boxplot(x='bin', y='y', data=df, color='grey', ax=ax, width=.6, fliersize=2)
        elif mode == 'violin':
            sns.violinplot(x='bin', y='y', data=df, color='grey', ax=ax, width=.6)
        elif mode == 'scatter':
            sns.scatterplot(x='bin', y='y', data=df)
        elif mode == 'raw':
            ax.plot(df.x, df.y, '.k', markersize=1)

        # set limits
        xlim = (-0.5, df.bin.max()+0.5)
        ax.set_xlim(*xlim)

        # format axes
        ax.set_xlabel('Red intensity')
        ax.set_ylabel('Green intensity')
        xticks = ax.get_xticks() * bin_size
        ax.set_xticklabels(['{:0.2f}'.format(s) for s in xticks])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # add model prediction to plot
        ax.plot(ax.get_xticks(), self.predict(xticks), '-r', linewidth=1)

        # store figure instance
        self.figs['fit'] = fig

    def show_correction(self, figsize=(6, 2), selected_only=False):
        """
        Show cell measurements before and after correction.

        Args:

            figsize (tuple) - figure size

            selected_only (bool) - if True, exclude cells outside selection bounds

        """

        # instantiate figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(nrows=1, ncols=2, wspace=.3)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        # add data to plots
        if selected_only:
            mask = self.layer.data.selected.values
        else:
            mask = np.ones(self.xt.size, dtype=bool)

        ax0.scatter(self.xt[mask], self.yt[mask], c='k', s=1, linewidth=0)
        ax1.scatter(self.xt[mask], self.ytc[mask], c='k', s=1, linewidth=0)

        # add model prediction to plot (dashed line for extrapolation)
        ax0.plot(self.xtdomain, self.predict(self.xtdomain), '--r', lw=1)
        ax0.plot(self.domain, self.predict(self.domain), '-r', lw=1.5)

        # label axes
        ax0.set_xlabel('Nuclear RFP level')
        ax0.set_ylabel('Nuclear GFP level')
        ax0.set_title('Original (Layer {:d})'.format(self.layer._id))
        ax1.set_ylabel('Corrected GFP level')
        ax1.set_xlabel('Nuclear RFP level')
        ax1.set_title('Corrected')

        # format axes
        xlim = (-0.02, self.xtdomain.max()+0.02)
        ylim = (-0.05, self.yt.max())
        for ax in (ax0, ax1):
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # store figure instance
        self.figs['correction'] = fig

    def show_background_extraction(self, **kwargs):
        """
        Visualize background extraction procedure.

        Returns:

            figure

        """
        bg_extraction = BackgroundExtraction(self.layer, niters=self.niters)
        return bg_extraction.plot_foreground_mask(**kwargs)

    def show_resampling(self, **kwargs):
        """
        Visualize resampling procedure.

        Returns:

            figures (tuple)

        """

        # extract background pixels
        bg_extraction = BackgroundExtraction(self.layer, niters=self.niters)
        bg_x0 = bg_extraction.extract_pixels('r')
        bg_y0 = bg_extraction.extract_pixels('g')

        # resample pixels
        bg_x, bg_y = resample_uniformly(bg_x0, bg_y0)

        # define plotting function
        def plot(bg_x, bg_y):
            """ Plot bg_x and bg_y distributions. """

            # create figure
            fig = plt.figure(figsize=(5, 1.25))
            gs = GridSpec(nrows=1, ncols=2, wspace=0.3)
            ax0 = fig.add_subplot(gs[0])
            ax1 = fig.add_subplot(gs[1])
            axes = (ax0, ax1)

            # plot pixel intensity distributions
            bins = np.arange(0, .4, .05)
            _ = axes[0].hist(bg_x, bins=bins, color='k')
            _ = axes[1].hist(bg_y, bins=bins, color='k')

            # format axes
            for ax in axes:
                ax.set_xlim(0, bins.max())
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                #ax.set_yticks([])
                ax.set_ylim(bottom=10, top=2e5)
                ax.set_yscale('log')
                ax.set_yticks([10, 100, 1000, 10000, 100000])
                ax.set_ylabel('No. pixels')
            ax0.set_xlabel('RFP level')
            ax1.set_xlabel('GFP level')

            return fig

        fig0 = plot(bg_x0, bg_y0)
        fig1 = plot(bg_x, bg_y)

        return fig0, fig1
