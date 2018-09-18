from scipy.ndimage import iterate_structure, generate_binary_structure
from scipy.ndimage import gaussian_filter, median_filter
from skimage.filters import threshold_otsu
from skimage.exposure import equalize_adapthist

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from .segmentation import Segmentation


class MonochromeImage:
    """
    Object represents a monochrome image.
    """

    def __init__(self, im, labels=None, channel=None):
        self.im = im
        self.shape = im.shape[:2]
        self.mask = np.ones_like(self.im, dtype=bool)
        self.labels = labels
        self.channel = channel

    def show(self,
             segments=True,
             cmap=None,
             vmin=0, vmax=1,
             figsize=(10, 10),
             ax=None,
             **kwargs):
        """
        Render image.

        Args:
        segments (bool) - if True, include cell segment contours
        cmap (matplotlib.colors.ColorMap)
        vmin, vmax (float) - bounds for color scale
        figsize (tuple) - figure size
        ax (matplotlib.axes.AxesSubplot) - if None, create axis
        kwargs: keyword arguments for add_contours

        Returns:
        fig (matplotlib.figures.Figure)
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        fig = plt.gcf()

        if cmap is None:
            if len(self.im.shape) == 2:
                im = self.im.reshape(*self.shape, 1)
                cpad = dict(r=(0, 2), g=(1, 1), b=(2, 0))
                pad = ((0,0), (0,0), cpad[self.channel])
                im = np.pad(im, pad, mode='constant', constant_values=0)
            else:
                im = self.im
            ax.imshow(im)

        else:
            ax.imshow(self.im, cmap=cmap, vmin=vmin, vmax=vmax)

        if segments and self.labels is not None:
            self.add_contours(ax, **kwargs)

        ax.axis('off')
        return fig

    def add_contour(self, ax, mask, lw=1, color='r'):
        """ Adds border of specified contour. """
        ax.contour(mask, [0.5], linewidths=[lw], colors=[color])

    def add_contours(self, ax, lw=1, color='r'):
        """ Adds borders of all contours. """
        mask = self.labels > 0
        ax.contour(mask, [0.5], linewidths=[lw], colors=[color])

    def gaussian_filter(self, sigma=(1., 1.)):
        self.im = gaussian_filter(self.im, sigma=sigma)

    def median_filter(self, radius=0, structure_dim=1):
        """ Convolves image with median filter. """
        struct = iterate_structure(generate_binary_structure(2, structure_dim), radius).astype(int)
        self.im = median_filter(self.im, footprint=struct)

    def set_mean_mask(self):
        """ Mask values below mean. """
        self.mask = np.zeros_like(self.im, dtype=bool)
        self.mask[self.im>=np.mean(self.im)] = True

    def set_otsu_mask(self):
        """ Mask values below otsu threahold. """
        threshold = threshold_otsu(self.im)
        self.mask = np.zeros_like(self.im, dtype=bool)
        self.mask[self.im>=threshold] = True

    def clahe(self, factor=8, clip_limit=0.01, nbins=256):
        """
        Runs CLAHE on reflection-padded image.

        Args:
        factor (float or int) - number of segments per dimension
        clip_limit (float) - clip limit for CLAHE
        nbins (int) - number of grey-scale bins for histogram
        """

        # set kernel size as fraction of image size
        kernel_size = [int(np.ceil(s/factor)) for s in self.im.shape]

        # pad image with reflection about boundaries (circumvents artefacts)
        im_padded = np.pad(self.im, [(x,)*2 for x in kernel_size], mode='reflect')

        # apply CLAHE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im_equalized = equalize_adapthist(im_padded, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins)

        # crop image
        self.im = im_equalized[tuple(slice(s, -s) for s in kernel_size)]

    def preprocess(self,
                   median_radius=2,
                   gaussian_sigma=(2, 2),
                   clip_limit=0.03,
                   clip_factor=20):
        """
        Preprocess image.
        """
        self.median_filter(radius=median_radius)
        self.gaussian_filter(sigma=gaussian_sigma)
        self.clahe(clip_limit=clip_limit, factor=clip_factor)

    def segment(self,
                preprocessing_kws={},
                seed_kws={},
                seg_kws={},
                min_segment_area=250):
        """
        Segment nuclear contours.
        """
        self.preprocess(**preprocessing_kws)
        segmentation = Segmentation(self, seed_kws=seed_kws, seg_kws=seg_kws)
        segmentation.exclude_small_segments(min_area=min_segment_area)
        self.labels = segmentation.labels


class MultichannelImage(MonochromeImage):
    """
    Object represents an RGB image.
    """

    def __init__(self, im, labels=None):
        """ Instantiate RGB image. """
        super().__init__(im, labels=labels)
        self.channels = dict(r=0, g=1, b=2)

    def get_channel(self, channel='b', copy=True):
        """ Returns monochrome image of specified color channel. """
        if copy:
            monochrome = deepcopy(self.im[:, :, self.channels[channel]])
        else:
            monochrome = self.im[:, :, self.channels[channel]]

        return MonochromeImage(monochrome, labels=self.labels, channel=channel)
