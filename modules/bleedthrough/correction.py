import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from modules.io import IO

from .models import GLM, OLS
from .resampling import resample_uniformly
from .background import BackgroundExtraction


class LayerCorrection(GLM):
    """
    Linear correction for background correlation between fluorescence channels.
    """

    def __init__(self, layer,
                 xvar='r',
                 yvar='g',
                 niters=10,
                 remove_zeros=False,
                 resample=False,
                 resample_size=None,
                 resample_cutoff=None,
                 seg_params=None,
                 **fit_kw):

        # store layer
        self.layer = layer

        # store parameters
        self.seg_params = seg_params
        self.xvar = xvar
        self.yvar = yvar
        self.niters = niters
        self.remove_zeros=remove_zeros,
        self.resample=resample,
        self.resample_size=resample_size,
        self.resample_cutoff=resample_cutoff,

        # extract X and Y pixels from background
        bg_ext = BackgroundExtraction(layer, niters, seg_params=seg_params)
        bg_x = bg_ext.extract_pixels(self.xvar)
        bg_y = bg_ext.extract_pixels(self.yvar)

        # remove zero-valued pixels
        if remove_zeros:
            bg_x, bg_y = self.remove_zeros(bg_x, bg_y)

        # resample uniformly in X
        if resample:
            bg_x, bg_y = resample_uniformly(bg_x, bg_y, resample_size, resample_cutoff)

        # fit line to background pixels
        super().__init__(bg_x, bg_y, **fit_kw)

        # instantiate container for figures
        self.figs = {}

    def correct_measurements(self):
        """
        Apply correction to measurements.
        """

        # store measurement values (test data)
        self.xt = layer.df[xvar].values
        self.yt = layer.df[yvar].values
        self.xtdomain = np.linspace(0, self.xt.max(), 10)

        # store model prediction and corrected measurements
        self.ytp = self.predict(self.xt)
        self.ytc = self.yt - self.ytp

    @staticmethod
    def remove_zeros(x, y):
        """ Remove pixels with zero values in either channel. """
        nonzero_mask = np.logical_and(x!=0, y!=0)
        return x[nonzero_mask], y[nonzero_mask]

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
        self.figs['background_fit'] = fig

    def show_correction(self, figsize=(6, 2), furrow_only=False):
        """
        Show cell measurements before and after correction.

        Args:
        figsize (tuple) - figure size
        furrow_only (bool) - if True, only include cells near the furrow
        """

        # instantiate figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(nrows=1, ncols=2, wspace=.3)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        # add data to plots
        if furrow_only:
            mask = self.layer.df.near_furrow.values
        else:
            mask = np.ones(self.x.size, dtype=bool)
        ax0.scatter(self.xt[mask], self.yt[mask], c='k', s=1, linewidth=0)
        ax1.scatter(self.xt[mask], self.ytc[mask], c='k', s=1, linewidth=0)

        # add model prediction to plot (dashed line for extrapolation)
        ax0.plot(self.xtdomain, self.predict(self.xtdomain), '--r', lw=1)
        ax0.plot(self.domain, self.predict(self.domain), '-r', lw=1.5)

        # label axes
        ax0.set_xlabel('Nuclear RFP level')
        ax0.set_ylabel('Nuclear GFP level')
        ax0.set_title('Original (Layer {:d})'.format(self.layer.layer_id))
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
        self.figs['background_correction'] = fig

    def save_figs(self, dpi=100, fmt='png'):
        """ Save all figures. """

        # keyword arguments for savefig
        kw = dict(dpi=dpi, format=fmt, transparent=True, rasterized=True)

        for name, fig in self.figs.items():

            # save figure
            path = os.path.join(self.layer.path, name+'.png')
            fig.savefig(path, **kw)

            # close figure
            fig.clf()
            plt.close(fig)
            gc.collect()

    def save(self, images=True):
        """
        Save linear model, corrected levels, and associated figures.

        Args:
        images (bool) - if True, save model fit and corrected measurement figs
        """

        # instantiate IO
        io = IO()

        # save metadata to json
        data = dict(mode=self.__class__.__name__,
                    xvar=self.xvar,
                    yvar=self.yvar,
                    seg_params=self.seg_params,
                    niters=self.niters,
                    remove_zeros=self.remove_zeros,
                    resample=self.resample,
                    resample_size=self.resample_size,
                    resample_cutoff=self.resample_cutoff,
                    coefficients=self.model.params.tolist())

        # write metadata to file
        io.write_json(os.path.join(self.layer.path, 'correction.json'), data)

        # save corrections for layer
        self.layer.df[self.yvar+'p'] = self.ytp
        self.layer.df[self.yvar+'_corrected'] = self.yt - self.ytp
        self.layer.save_contours()

        # save figures
        if images:
            self.show_fit()
            self.show_correction()
            self.save_figs()


class StackCorrection:
    """
    Linear correction for background correlation between fluorescence channels, applied to entire image stack.
    """

    def __init__(self, stack, **kw):

        # load segmentation params
        self.stack = stack
        self.seg = stack.load_metadata()['params']['segmentation_kw']

        # instantiate corrections
        self.corrections = {}
        self.correct(**kw)

    @staticmethod
    def load(path):
        """ USE STORED PARAMETERS TO AVOID OVERWRITING *TO DO* """
        pass

    def correct(self, **kw):
        """ Correct all included layers in stack. """
        for layer in self.stack:
            if layer.include:
                self.correct_layer(layer, **kw)

    def correct_layer(self, layer, **kw):
        """ Correct individual layer. """
        correction = LayerCorrection(layer, seg_params=self.seg, **kw)
        correction.correct_measurements()
        self.corrections[layer_id] = correction

    def show(self):
        """ Show all corrections. """
        for layer_id, correction in self.corrections.items():
            correction.show_correction()

    def save(self):
        """ Save all corrections. """
        for correction in self.corrections.values():
            correction.save()
