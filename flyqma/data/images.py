from scipy.ndimage import iterate_structure, generate_binary_structure
from scipy.ndimage import gaussian_filter, median_filter
from skimage.filters import threshold_otsu
from skimage.exposure import equalize_adapthist
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import warnings


class ImageScalar:
    """
    Object represents a monochrome image.

    Attributes:

        im (np.ndarray[float]) - 2D array of pixel values

        shape (array like) - image dimensions

        mask (np.ndarray[bool]) - image mask

        labels (np.ndarray[int]) - segment ID mask

    """

    def __init__(self, im, labels=None):
        """
        Instantiate scalar image.

        Args:

            im (np.ndarray[float]) - 2D array of pixel values

           labels (np.ndarray[int]) - segment ID mask

        """
        self.im = im
        self.shape = im.shape[:2]
        self.mask = np.ones_like(self.im, dtype=bool)
        self.labels = labels

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

            cmap (matplotlib.colors.ColorMap or str) - colormap or RGB channel

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

        # show image in RGB format
        if type(cmap) == str:
            ind = 'rgb'.index(cmap)
            im = np.zeros((self.shape[0], self.shape[1], 3))
            im[:, :, ind] = self.im
            ax.imshow(im)

        # otherwise use specified colormap
        elif cmap is not None:
            ax.imshow(self.im, cmap=cmap, vmin=vmin, vmax=vmax)

        # otherwise show raw image
        else:
            ax.imshow(self.im)

        # add segment labels
        if segments and self.labels is not None:
            self.add_contours(ax, **kwargs)

        # remove axis
        ax.axis('off')

        return fig

    def add_contour(self, ax, mask, lw=1, color='r'):
        """ Adds border of specified contour. """
        ax.contour(mask, [0.5], linewidths=[lw], colors=[color])

    def add_contours(self, ax, lw=1, color='r', rasterized=False):
        """ Adds borders of all contours. """
        mask = self.labels > 0
        ctr = ax.contour(mask, [0.5], linewidths=[lw], colors=[color])
        if rasterized:
            for c in ctr.collections:
                c.set_rasterized(True)

    def gaussian_filter(self, sigma=(1., 1.)):
        """ Apply 2D gaussian filter. """
        self.im = gaussian_filter(self.im, sigma=sigma)

    def median_filter(self, radius=0, structure_dim=1):
        """ Apply 2D median filter. """
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
        Run CLAHE on reflection-padded image.

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

        Args:

            median_radius (int) - median filter size, px

            gaussian_sigma (tuple) - gaussian filter size, px std dev

            clip_limit (float) - CLAHE clip limit

            clip_factor (int) - CLAHE clip factor

        """
        self.median_filter(radius=median_radius)
        self.gaussian_filter(sigma=gaussian_sigma)
        self.clahe(clip_limit=clip_limit, factor=clip_factor)


class ImageMultichromatic(ImageScalar):
    """
    Object represents a multichromatic image.

    Attributes:

        im (np.ndarray[float]) - 2D array of pixel values in WHC format

    Inherited attributes:

        shape (array like) - image dimensions

        mask (np.ndarray[bool]) - image mask

        labels (np.ndarray[int]) - segment ID mask

    """

    def __init__(self, im, labels=None):
        """
        Instantiate RGB image.

        Args:

            im (np.ndarray[float]) - 2D array of pixel values

           labels (np.ndarray[int]) - segment ID mask

        """
        super().__init__(im, labels=labels)

    def get_channel(self, channel, copy=True):
        """
        Returns monochrome image of specified color channel.

        Args:

            channel (int) - desired channel

            copy (bool) - if True, instantiate from image copy

        Returns:

            image (ImageScalar) - monochrome image

        """

        if type(channel) == str:
            if 'ch' in channel:
                channel = int(channel.strip('ch'))

        if copy:
            monochrome = deepcopy(self.im[:, :, channel])
        else:
            monochrome = self.im[:, :, channel]

        return ImageScalar(monochrome, labels=self.labels)

    def to_RGB(self, channels_dict=None, copy=True):
        """
        Returns RGB image of specified color channels.

        Args:

            channels_dict (dict) - RGB channels keyed by channel index

            copy (bool) - if True, instantiate from image copy

        Returns:

            image (ImageMultichromatic) - RGB image

        """

        # default to first three channels as R, G, and B
        if channels_dict is None:
            channels_dict = dict(enumerate('rgb'))

        # copy image
        if copy:
            im = deepcopy(self.im)
        else:
            im = self.im

        # concatenate channels
        lookup = {v.lower().strip(): k for k, v in channels_dict.items()}
        channels = []
        for channel in 'rgb':
            if channel in lookup.keys():
                idx = lookup[channel]
                channels.append(im[:, :, idx])
            else:
                channels.append(np.zeros_like(im[:,:,0]))
        rgb = np.stack(channels, axis=-1)

        return ImageMultichromatic(rgb, labels=self.labels)
