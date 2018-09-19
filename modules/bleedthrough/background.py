
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, generate_binary_structure
from matplotlib.gridspec import GridSpec


class BackgroundExtraction:
    """
    Object for extracting image background pixels.
    """

    def __init__(self, layer, niters=10):

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

    def isolate_pixels(self, channel='r'):
        """
        Isolate pixels in image background.

        Args:
        channel (str) - channel to be extracted, 'r', 'g', or 'b'

        Returns:
        px (np.ma.masked_array) - background masked pixel intensities
        """
        px = self.layer.get_channel(channel).im
        return np.ma.masked_array(px, ~self.bg_mask)

    def extract_pixels(self, channel='r'):
        """
        Extract pixels from image background.

        Args:
        channel (str) - channel to be extracted, 'r', 'g', or 'b'

        Returns:
        px (np.ndarray[float]) - 1D array of background pixel intensities
        """
        bg_px = self.isolate_pixels(channel)
        return bg_px[~bg_px.mask].data

    # def show_mask(self, cmap=plt.cm.viridis):
    #     """ Show background pixels. """
    #     fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6))
    #     cmap.set_bad('w')
    #     ax0.imshow(self.xx, cmap=cmap, vmin=0, vmax=1)
    #     ax1.imshow(self.yy, cmap=cmap, vmin=0, vmax=1)
    #     for ax in (ax0, ax1):
    #         ax.axis('off')
    #     plt.tight_layout()
    #     return fig

    # def show_pixel_distributions(self, original=False):
    #     """ Plot distribution of background pixel values. """

    #     if original:
    #         bg_x = self.xx[~self.xx.mask].data
    #         bg_y = self.yy[~self.yy.mask].data
    #     else:
    #         bg_x, bg_y = self.bg_x, self.bg_y

    #     # instantiate figure

    #     fig = plt.figure(figsize=(4, 1))
    #     gs = GridSpec(nrows=1, ncols=2, wspace=.3)
    #     ax0 = plt.subplot(gs[0])
    #     ax1 = plt.subplot(gs[1])

    #     xmax = np.unique(self.xdomain)[-2]

    #     bins = np.linspace(0, xmax, 12)
    #     _ = ax0.hist(bg_x, bins=bins, density=False, color='k')
    #     _ = ax1.hist(bg_y, bins=bins, density=False, color='k')
    #     ax0.set_xlabel('Nuclear RFP level', fontsize=7)
    #     ax1.set_xlabel('Nuclear GFP level', fontsize=7)
    #     ax0.set_ylabel('Frequency', fontsize=7, labelpad=2)

    #     # format axes
    #     xlim = (-0.02, xmax+0.02)
    #     for ax in (ax0, ax1):
    #         ax.set_yticks([])
    #         ax.set_xlim(*xlim)
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['left'].set_visible(False)
    #         ax.spines['right'].set_visible(False)

    #     return fig

