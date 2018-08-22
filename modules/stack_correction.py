
import os
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, generate_binary_structure

import statsmodels.api as sm
from collections import Counter

from modules.formatting import *
from modules.io import IO

import seaborn as sns
import warnings




class BackgroundExtraction:
    def __init__(self, layer, niters=10, **kw):

        self.layer = layer

        # set background
        self.niters = niters
        self.bg_mask = self.build_background_mask(niters, **kw)

        # get background pixels
        xx, yy = self.extract_pixels(layer, self.bg_mask)
        self.xx = xx
        self.yy = yy
        bg_x, bg_y = xx[~xx.mask].data, yy[~yy.mask].data
        self.bg_x, self.bg_y = bg_x, bg_y

    def build_background_mask(self, niters, **kw):
        bg = self.layer.get_channel('b')
        _ = bg.preprocess(**kw)
        bg.set_otsu_mask()
        bg_mask = bg.mask
        bg_mask = self.dilate_foreground(bg.mask, niters)
        return bg_mask

    @staticmethod
    def dilate_foreground(fg, niters=5):
        struct = generate_binary_structure(2, 2)
        fg_mask = binary_dilation(fg, struct, niters)
        bg_mask = ~fg_mask
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


class StackExtraction:
    def __init__(self, disc_id, stack, layers, **kw):
        disc_id = str(disc_id)
        for layer_id in layers:
            layer = stack.get_layer(layer_id)
            extraction = BackgroundExtraction(layer, **kw)
            p = os.path.join('./stack_backgrounds', disc_id, str(layer_id))
            self.save(extraction, p)

    def save(self, extraction, path):
        data = dict(r=extraction.bg_x.tolist(), g=extraction.bg_y.tolist())
        io = IO()
        io.write_json(path+'.json', data)




class OLS:

    def __init__(self, x, y, resample=True, cutoff=None, **fit_kw):
        x, y = self.filter_pixels(x, y, resample=resample, cutoff=cutoff)
        self.x = x
        self.y = y
        self.model = self.fit(x, y, **fit_kw)

    def save(self, path):

        # save parameters
        io = IO()
        params = self.model.params.tolist()
        io.write_json(os.path.join(path, 'line.json'), params)

        # save figure
        fig = self.show_fit()
        fig_kw = dict(format='pdf', rasterized=True, transparent=True, dpi=100)
        fig.savefig(os.path.join(path, 'fit.pdf'), **fig_kw)
        fig.clf()
        plt.close(fig)

    def __call__(self, x):
        return self.predict(x)

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

    @classmethod
    def filter_pixels(cls, x, y,
                      resample=False,
                      size=None,
                      cutoff=None):
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

    def show_fit(self, mode='box', bin_size=0.05, figsize=(3, 2), showfliers=False):
        """ Plot fit to background pixels. """

        # intantiate figure
        fig, ax = plt.subplots(figsize=figsize)

        # compile dataframe
        xy = np.vstack((self.x, self.y)).T
        df = pd.DataFrame.from_records(xy, columns=['x', 'y'])

        # add data to plot
        df['bin'] = (df.x // bin_size)
        if mode == 'strip':
            sns.stripplot(x='bin', y='y', data=df, ax=ax, size=1, color='k')
        elif mode == 'box':
            sns.boxplot(x='bin', y='y', data=df, color='grey', ax=ax, width=.6, fliersize=2, showfliers=showfliers)
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
        ax.set_xticklabels(['{:0.1f}'.format(s) for s in xticks])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # add model prediction to plot
        ax.plot(ax.get_xticks(), self.predict(xticks), '-r', linewidth=1)

        return fig


class GLM(OLS):
    """
    GLM model.
    """

    def fit(self, x, y, N=30000, maxiter=100, shift=0):
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





