from os.path import join
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..utilities import IO
from ..visualization import *

from .models import GLM
from .resampling import resample_uniformly
from .background import BackgroundExtraction
from .visualization import CorrectionVisualization
from .visualization import LayerCorrectionVisualization


class Correction(GLM, CorrectionVisualization):
    """
    Linear correction for background correlation between fluorescence channels within an individual layer.

    Attributes:

        xt, yt (np.ndarray[float]) - foreground measurements

        xraw, yraw (np.ndarray[float]) - raw background pixel intensities

        x, y (np.ndarray[float]) - resampled background pixel intensities

    """

    def __init__(self, xt, yt, bg_x, bg_y,
                 remove_zeros=False,
                 resample=True,
                 resample_size=None,
                 resample_cutoff=None,
                 store_pixels=True,
                 **fit_kw):
        """
        Instantiate bleedthrough correction for an RGB image layer.

        Args:

            data (pd.DataFrame) - measurement data

            xt, yt (np.ndarray[float]) - foreground measurements

            bg_x, bg_y (np.ndarray[float]) - background pixel intensities

            remove_zeros (bool) - if True, remove all zero-valued pixels.

            resample (bool) - if True, uniformly resample pixels in X

            resample_size (int) - number of uniform samples

            resample_cutoff (int) - upper bound for samples (quantile, 0-100)

            store_pixels (bool) - if True, store raw background pixels

        """

        # store data
        self.xt = xt
        self.yt = yt

        self.store_pixels = store_pixels
        if store_pixels:
            self.xraw = bg_x
            self.yraw = bg_y

        # store parameters
        self.remove_zeros = remove_zeros
        self.resample = resample
        self.resample_size = resample_size
        self.resample_cutoff = resample_cutoff

        # remove zero-valued pixels
        if remove_zeros:
            bg_x, bg_y = self._remove_zeros(bg_x, bg_y)

        # resample uniformly in X
        if resample:
            bg_x, bg_y = resample_uniformly(bg_x, bg_y, resample_size, resample_cutoff)

        # fit line to background pixels
        super().__init__(bg_x, bg_y, **fit_kw)

        # apply correction to measurements (internally)
        self.correct_measurements()

        # instantiate container for figures
        self.figs = {}

    def correct_measurements(self):
        """ Apply correction to measurements. """

        # store measurement values (test data)
        self.xtdomain = np.linspace(0, self.xt.max(), 10)

        # store model prediction and corrected measurements
        self.ytp = self.predict(self.xt)
        self.ytc = self.yt - self.ytp

    @staticmethod
    def _remove_zeros(x, y):
        """ Remove pixels with zero values in either channel. """
        nonzero_mask = np.logical_and(x!=0, y!=0)
        return x[nonzero_mask], y[nonzero_mask]


class LayerCorrection(Correction, LayerCorrectionVisualization):
    """
    Linear correction for background correlation between fluorescence channels within an individual layer.

    Attributes:

        layer (Layer) - layer RGB image

    Inherited attributes:

        xt, yt (np.ndarray[float]) - foreground measurements

        xraw, yraw (np.ndarray[float]) - raw background pixel intensities

        x, y (np.ndarray[float]) - resampled background pixel intensities

    Parameters:

        xvar (str) - name of independent variable attribute in measurement data

        yvar (str) - name of dependent variable attribute in measurement data

        niters (int) - number of binary dilations applied to foreground mask

        remove_zeros (bool) - if True, remove all zero-valued pixels.

        resample (bool) - if True, uniformly resample pixels in X

        resample_size (int) - number of uniform samples

        resample_cutoff (int) - upper bound for samples (quantile, 0 to 100)

    """

    def __init__(self, layer, xvar, yvar,
                 niters=50,
                 remove_zeros=False,
                 resample=True,
                 resample_size=None,
                 resample_cutoff=None,
                 store_pixels=False,
                 **fit_kw):
        """
        Instantiate bleedthrough correction for an RGB image layer.

        Args:

            layer (Layer) - layer RGB image

            xvar (int) - independent color channel

            yvar (int) - dependent color channel

            niters (int) - number of binary dilations applied to foreground

            remove_zeros (bool) - if True, remove all zero-valued pixels.

            resample (bool) - if True, uniformly resample pixels in X

            resample_size (int) - number of uniform samples

            resample_cutoff (int) - upper bound for samples (quantile, 0-100)

            store_pixels (bool) - if True, store raw background pixels

        """

        # store layer
        self.layer = layer
        self.xvar = xvar
        self.yvar = yvar
        self.niters = niters

        # get foreground measurements
        xt = layer.data[self.xkey].values
        yt = layer.data[self.ykey].values

        # extract X and Y pixels from background
        bg_x, bg_y = self.extract_background()

        # instantiate correction
        super().__init__(xt, yt, bg_x, bg_y,
                         remove_zeros=remove_zeros,
                         resample=resample,
                         resample_size=resample_size,
                         resample_cutoff=resample_cutoff,
                         store_pixels=store_pixels,
                         **fit_kw)

    @property
    def xkey(self):
        """ DataFrame key for independent channel. """
        return 'ch{:d}'.format(self.xvar)

    @property
    def ykey(self):
        """ DataFrame key for dependent channel. """
        return 'ch{:d}'.format(self.yvar)

    def extract_background(self):
        """ Returns raw background pixels. """
        bg_ext = BackgroundExtraction(self.layer, self.niters)
        bg_x = bg_ext.extract_pixels(self.xvar)
        bg_y = bg_ext.extract_pixels(self.yvar)
        return bg_x, bg_y

    @classmethod
    def load(cls, layer):
        """
        Load linear model from file.

        Args:

            path (str) - path to correction directory

        Returns:

            correction (LayerCorrection)

        """

        path = layer.subdirs['correction']

        # load data
        io = IO()
        data = io.read_json(join(path, 'data.json'))

        return LayerCorrection(layer, **data['params'])

    def save(self, images=True):
        """
        Save linear model and corrected levels.

        Args:

            images (bool) - if True, save model fit and corrected measurement figs

        """

        # add subdirectory to layer
        self.layer.make_subdir('correction')
        path = self.layer.subdirs['correction']

        # instantiate IO
        io = IO()

        # save metadata to json
        params = dict(
                    xvar=self.xvar,
                    yvar=self.yvar,
                    niters=self.niters,
                    remove_zeros=self.remove_zeros,
                    resample=self.resample,
                    resample_size=self.resample_size,
                    resample_cutoff=self.resample_cutoff)

        data = dict(mode=self.__class__.__name__,
                    params=params,
                    coefficients=self.model.params.tolist())

        # write metadata to file
        io.write_json(join(path, 'data.json'), data)

        # update measurements
        self.layer.apply_correction(self.layer.data)
        self.layer.save_processed_data()

        # save figures
        if images:
            self.show_fit()
            self.show_correction()
            self.save_figs()

    def save_figs(self, dpi=100, fmt='png'):
        """
        Save all figures.

        Args:

            dpi (int) - resolution

            fmt (str) - image format

        """

        # get correction path
        dirpath = self.layer.subdirs['correction']

        # keyword arguments for savefig
        kw = dict(dpi=dpi, format=fmt, transparent=True, rasterized=True)

        for name, fig in self.figs.items():

            # save figure
            path = join(dirpath, name+'.png')
            fig.savefig(path, **kw)

            # close figure
            fig.clf()
            plt.close(fig)
            gc.collect()
