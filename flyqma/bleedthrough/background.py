from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, generate_binary_structure
from matplotlib.gridspec import GridSpec

from ..visualization.masking import Mask


class BackgroundExtraction:
    """
    Object for extracting image background pixels.

    Attributes:

        layer (Layer) - layer RGB image

        bg_mask (np.ndarray[bool]) - background mask, True where background

    """

    def __init__(self, layer, niters=10):
        """
        Instantiate background extraction.

        Args:

            layer (Layer) - layer RGB image

            niters (int) - number of binary foreground dilations

        """

        # store layer
        self.layer = layer

        # build background mask
        self.bg_mask = self.build_background_mask(niters)

    def build_background_mask(self, niters=0):
        """
        Construct background mask by dilating foregound.

        Args:

            niters (int) - number of binary dilations

        Returns:

            bg_mask (np.ndarray[bool]) - background mask, True where background

        """

        # re-run image preprocessing to obtain foreground threshold
        seg_params = self.layer.metadata['params']['segmentation_kw']
        preprocessing_kws = seg_params['preprocessing_kws']
        if preprocessing_kws is not None:
            bg = self.layer.get_channel(self.layer.metadata['bg'])
            _ = bg.preprocess(**preprocessing_kws)
            bg.set_otsu_mask()
            bg_mask = bg.mask
        else:
            bg_mask = (self.layer.labels!=0)

        # dilate foreground
        bg_mask = self.dilate_foreground(bg.mask, niters)

        return bg_mask

    @staticmethod
    def dilate_foreground(foreground, niters=5):
        """ Apply binary dilation to foreground mask. """
        struct = generate_binary_structure(2, 2)
        fg_mask = binary_dilation(foreground, struct, niters)
        bg_mask = ~fg_mask
        return bg_mask

    def isolate_pixels(self, channel):
        """
        Isolate pixels in image background.

        Args:

            channel (int) - color channel to be extracted

        Returns:

           px (np.ma.masked_array) - background masked pixel intensities

        """
        px = self.layer.get_channel(channel).im
        return np.ma.masked_array(px, ~self.bg_mask)

    def extract_pixels(self, channel):
        """
        Extract pixels from image background.

        Args:

            channel (int) - color channel to be extracted

        Returns:

            px (np.ndarray[float]) - 1D array of background pixel intensities

        """
        bg_px = self.isolate_pixels(channel)
        return bg_px[~bg_px.mask].data

    def plot_foreground_mask(self, invert=False, ax=None, figsize=(3, 3)):
        """
        Plot foreground mask.

        Args:

            invert (bool) - if True, mask background rather than foreground

            ax (matplotlib.axes.AxesSubplot) - if None, create figure

            figsize (tuple) - figure size

        Returns:

            figure

        """

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()

        # TO DO: handle alternate channel specification... non RGB... etc
        # extract and visualize red/green channels
        rg = deepcopy(self.layer.im)
        rg[:,:,-1] = 0
        ax.imshow(rg)

        # add foreground mask
        if invert:
            mask = Mask(~self.bg_mask)
        else:
            mask = Mask(self.bg_mask)

        mask.add_contourf(ax, alpha=0.5, hatches=['//'])
        mask.add_contour(ax, lw=2, color='w')
        ax.axis('off')

        return fig
