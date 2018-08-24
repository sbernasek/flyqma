import os
import warnings
import gc
#warnings.filterwarnings('error')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, generate_binary_structure

import statsmodels.api as sm
from collections import Counter

from modules.formatting import *
from modules.io import IO


class LinearCorrection:
    """
    Linear correction for background correlation between fluorescence channels.
    """

    def __init__(self, layer,
                 xvar='r', yvar='g',
                 niters=10,
                 xmax=1, ymax=1,
                 remove_zeros=False,
                 resample=False,
                 resample_size=None,
                 resample_cutoff=None,
                 seg_params=None,
                 apply_correction=True,
                 **fit_kw):

        # store layer
        self.layer = layer

        # store parameters
        self.seg_params = seg_params
        self.xvar = xvar
        self.yvar = yvar
        self.niters = niters
        self.xmax = xmax
        self.ymax = ymax
        self.remove_zeros=remove_zeros,
        self.resample=resample,
        self.resample_size=resample_size,
        self.resample_cutoff=resample_cutoff,

        # set background
        self.bg_mask = self.build_background_mask(niters)

        # get background pixels
        xx, yy = self.extract_pixels(layer, self.bg_mask)
        self.xx = xx
        self.yy = yy
        bg_x, bg_y = xx[~xx.mask].data, yy[~yy.mask].data

        # filter pixels
        bg_x, bg_y = self.filter_pixels(bg_x, bg_y,
                           remove_zeros=remove_zeros,
                           resample=resample,
                           size=resample_size,
                           cutoff=resample_cutoff)
        self.bg_x, self.bg_y = bg_x, bg_y

        # fit line to background pixels
        self.model = self.fit(self.bg_x, self.bg_y, **fit_kw)

        if apply_correction:

            # store measurement values
            self.x = layer.df[xvar].values
            self.y = layer.df[yvar].values
            self.xdomain = np.linspace(0, self.x.max(), 10)

            # store model prediction and corrected measurements
            self.yp = self.predict(self.x)
            self.yc = self.__call__(self.x, self.y)

        # instantiate container for figures
        self.figs = {}

    def __call__(self, x, y):
        yp = self.predict(x)
        return y - yp

    def fit(self, x, y, **kw):
        """ Fit model. """
        self.domain = np.linspace(0, x.max(), 10)
        x = sm.tools.add_constant(x.reshape(-1, 1))
        model = sm.OLS(y, x, hasconst=None).fit()
        return model

    def predict(self, x):
        """ Make model prediction. """
        xx = sm.tools.add_constant(x.reshape(-1, 1))
        return self.model.predict(xx)

    @staticmethod
    def dilate_foreground(fg, niters=5):
        struct = generate_binary_structure(2, 2)
        fg_mask = binary_dilation(fg, struct, niters)
        bg_mask = ~fg_mask
        return bg_mask

    def build_background_mask(self, niters):

        if self.seg_params is not None:
            bg = self.layer.get_channel('b')
            _ = bg.preprocess(**self.seg_params['preprocessing_kws'])
            bg.set_otsu_mask()
            bg_mask = bg.mask
        else:
            bg_mask = (self.layer.labels!=0)

        bg_mask = self.dilate_foreground(bg.mask, niters)
        return bg_mask

    @classmethod
    def extract_pixels(cls, layer, bg_mask,
                          xvar='r', yvar='g',
                          xmax=1, ymax=1):

        # extract pixels
        bg_x = layer.get_channel(xvar).im
        bg_y = layer.get_channel(yvar).im

        # mask pixels below threshold intensity
        intensity_mask = (bg_x<=xmax) & (bg_y<=ymax)
        mask = np.logical_and(intensity_mask, bg_mask)
        xx = np.ma.masked_array(bg_x, ~mask)
        yy = np.ma.masked_array(bg_y, ~mask)

        return xx, yy

    @classmethod
    def filter_pixels(cls, x, y,
                      remove_zeros=False,
                      resample=False,
                      size=None,
                      cutoff=None):

        # remove x zeros
        if remove_zeros:
            nonzero_mask = (x != 0)
            nonzero_mask = np.logical_and(nonzero_mask, y!=0)
            x, y = x[nonzero_mask], y[nonzero_mask]

        # uniformly resample in x
        if resample:
            x, y = cls.resample(x, y, size=size, cutoff=cutoff)

        return x, y

    @staticmethod
    def resample(x, y, size=None, cutoff=None):
        """ Resample uniformly in X. """

        if size is None:
            size = x.size

        # sort values
        sort_ind = np.argsort(x)
        xx, yy = x[sort_ind], y[sort_ind]

        # apply threshold on upper bound
        if cutoff is not None:
            threshold = np.percentile(xx, cutoff)
        else:
            threshold = xx.max()+1

        # get unique x values
        xunique = np.unique(xx)

        # filter points below threshold
        below_threshold = (xx<=threshold)
        xx, yy = xx[below_threshold], yy[below_threshold]

        # get probabilities
        x_to_count = np.vectorize(Counter(xx).get)

        # get intervals
        intervals = np.diff(xunique)
        unique_below_threshold = (xunique[:-1]<=threshold)
        intervals = intervals[unique_below_threshold]

        # assign probabilities
        x_to_cumul = np.vectorize(dict(zip(xunique[:-1][unique_below_threshold], intervals/intervals.sum())).get)
        p = x_to_cumul(xx)/x_to_count(xx)
        p[np.isnan(p)] = 0

        # generate sample
        sample_ind = np.random.choice(np.arange(xx.size), size=size, p=p)
        xu, yu = xx[sample_ind], yy[sample_ind]

        return xu, yu

    def show_mask(self, cmap=plt.cm.viridis):
        """ Show background mixels. """
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6))
        cmap.set_bad('w')
        ax0.imshow(self.xx, cmap=cmap, vmin=0, vmax=1)
        ax1.imshow(self.yy, cmap=cmap, vmin=0, vmax=1)
        for ax in (ax0, ax1):
            ax.axis('off')
        plt.tight_layout()
        return fig

    def show_pixel_distributions(self, original=False):
        """ Plot distribution of background pixel values. """

        if original:
            bg_x = self.xx[~self.xx.mask].data
            bg_y = self.yy[~self.yy.mask].data
        else:
            bg_x, bg_y = self.bg_x, self.bg_y

        # instantiate figure

        fig = plt.figure(figsize=(4, 1))
        gs = GridSpec(nrows=1, ncols=2, wspace=.3)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        xmax = np.unique(self.xdomain)[-2]

        bins = np.linspace(0, xmax, 12)
        _ = ax0.hist(bg_x, bins=bins, density=False, color='k')
        _ = ax1.hist(bg_y, bins=bins, density=False, color='k')
        ax0.set_xlabel('Nuclear RFP level', fontsize=7)
        ax1.set_xlabel('Nuclear GFP level', fontsize=7)
        ax0.set_ylabel('Frequency', fontsize=7, labelpad=2)

        # format axes
        xlim = (-0.02, xmax+0.02)
        for ax in (ax0, ax1):
            ax.set_yticks([])
            ax.set_xlim(*xlim)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

        return fig

    def show_fit(self, mode='box', bin_size=0.05, figsize=(3, 2)):
        """ Plot fit to background pixels. """

        # intantiate figure
        fig, ax = plt.subplots(figsize=figsize)

        # compile dataframe
        bg_xy = np.vstack((self.bg_x, self.bg_y)).T
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
        """ Show cell measurements before and after correction. """

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
        ax0.scatter(self.x[mask], self.y[mask], c='k', s=1, linewidth=0)
        ax1.scatter(self.x[mask], self.yc[mask], c='k', s=1, linewidth=0)

        # add model prediction to plot (dashed line for extrapolation)
        ax0.plot(self.xdomain, self.predict(self.xdomain), '--r', linewidth=1)
        ax0.plot(self.domain, self.predict(self.domain), '-r', linewidth=1.5)

        # label axes
        ax0.set_xlabel('Nuclear RFP level')
        ax0.set_ylabel('Nuclear GFP level')
        ax0.set_title('Original (Layer {:d})'.format(self.layer.layer_id))
        ax1.set_ylabel('Corrected GFP level')
        ax1.set_xlabel('Nuclear RFP level')
        ax1.set_title('Corrected')

        # format axes
        xlim = (-0.02, self.xdomain.max()+0.02)
        ylim = (-0.05, self.y.max())
        for ax in (ax0, ax1):
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # store figure instance
        self.figs['background_correction'] = fig

    def save_figs(self):
        """ Save all figures. """
        kw = dict(dpi=100, format='png', transparent=True, rasterized=True)
        for name, fig in self.figs.items():

            # save figure
            path = os.path.join(self.layer.path, name+'.png')
            fig.savefig(path, **kw)

            # close figure
            fig.clf()
            plt.close(fig)
            gc.collect()

    def save(self):

        # instantiate IO
        io = IO()

        # save metadata to json
        data = dict(mode=self.__class__.__name__,
                    xvar=self.xvar,
                    yvar=self.yvar,
                    seg_params=self.seg_params,
                    niters=self.niters,
                    xmax=self.xmax,
                    ymax=self.ymax,
                    remove_zeros=self.remove_zeros,
                    resample=self.resample,
                    resample_size=self.resample_size,
                    resample_cutoff=self.resample_cutoff,
                    coefficients=self.model.params.tolist())

        # write metadata to file
        io.write_json(os.path.join(self.layer.path, 'correction.json'), data)

        # save corrections for layer
        self.layer.df[self.yvar+'p'] = self.yp
        self.layer.df[self.yvar+'_corrected'] = self.y - self.yp
        self.layer.save_contours()

        # save figures
        self.show_fit()
        self.show_correction()
        self.save_figs()


class GLMCorrection(LinearCorrection):
    """
    Linear correction for background correlation between fluorescence channels.
    """

    def fit(self, x, y, N=10000, maxiter=100, shift=0):
        """ Fit Gamma GLM with identity link function. """

        self.domain = np.linspace(0, x.max(), 10)

        # downsample
        if N is not None:
            ind = np.random.randint(0, x.size, size=N)
            x, y = x[ind], y[ind]

        # construct variables
        xx = sm.tools.add_constant(x.reshape(-1, 1))
        yy = y + shift

        # define model
        family = sm.families.Gamma(link=sm.families.links.identity())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glm = sm.genmod.GLM(yy, xx, family=family)

        # fit model
        start_params = [0.1+shift, 0.5]
        model = glm.fit(start_params=start_params, maxiter=maxiter)
        if not model.converged:
            raise Warning('GLM did not converge.')

        return model


class StackCorrector:

    def __init__(self, stack, **kw):

        # load segmentation params
        self.stack = stack
        self.seg = stack.load_metadata()['params']['segmentation_kw']

        # instantiate corrections
        self.corrections = {}
        self.instantiate(**kw)

    @staticmethod
    def load(path):
        """ USE STORED PARAMETERS TO AVOID OVERWRITING *TO DO* """
        pass

    def instantiate(self, **kw):
        for layer in self.stack:
            if layer.include:
                correction = self._correct(layer, self.seg, **kw)
                self.corrections[layer.layer_id] = correction

    def correct(self, layer_id, **kw):
        layer = self.stack[layer_id]
        self.corrections[layer_id] = self._correct(layer, self.seg, **kw)

    @staticmethod
    def _correct(layer, seg, mode='glm', **kw):
        if mode == 'glm':
            correction = GLMCorrection(layer, seg_params=seg, **kw)
        else:
            correction = LinearCorrection(layer, seg_params=seg, **kw)
        return correction

    def show(self):
        """ Show all corrections. """
        for layer_id, correction in self.corrections.items():
            correction.show_correction()

    def save(self):
        """ Save all corrections. """
        for correction in self.corrections.values():
            correction.save()






