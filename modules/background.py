from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np
from modules.io import IO


class BackgroundCorrection:
    """
    Linear correction for background correlation between fluorescence channels.
    """

    def __init__(self, layer, xvar='r', yvar='g', niters=10):

        # set background
        self.xvar = xvar
        self.yvar = yvar
        self.niters = niters
        self.bg_mask = self.build_bg_mask(layer, niters)
        bg_x = layer.get_channel(xvar).im[self.bg_mask]
        bg_y = layer.get_channel(yvar).im[self.bg_mask]

        # fit line
        m, b = self.fit_line(bg_x, bg_y)
        self.m = m
        self.b = b

        # apply correction
        self.x = layer.df[xvar].values
        self.y = layer.df[yvar].values
        self.yp = self.m*self.x + self.b
        self.yc = self.__call__(self.x, self.y)

    # def __call__(self, x, y):
    #     return (y / (self.m*x + self.b)) * self.b

    def __call__(self, x, y):
        yp = (self.m*x + self.b)
        scaling = np.mean(y)
        return (((y - yp)/yp) * scaling) + self.b

    # def __call__(self, x, y):
    #     yp = (self.m*x + self.b)
    #     return y - yp

    def save(self, dirpath):

        # instantiate IO
        io = IO()

        # save metadata to json
        params = dict(xvar=self.xvar, yvar=self.yvar, niters=self.niters)
        fit = dict(m=self.m, b=self.b)
        data = dict(params=params, fit=fit)
        io.write_json(os.path.join(dirpath, 'bg_correction.json'), data)

    @staticmethod
    def build_bg_mask(layer, niters=5):
        struct = generate_binary_structure(2, 2)
        fg_mask = binary_dilation(layer.labels!=0, struct, niters)
        bg_mask = ~fg_mask
        return bg_mask

    @staticmethod
    def fit_line(x, y):
        reg = linregress(x, y)
        m, b = reg.slope, reg.intercept
        return m, b

    def show_bg_mask(self):
        """ Show background mask. """
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(self.bg_mask)

    def show_correction(self):
        fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, figsize=(8, 3))

        # plot raw data with fit
        ax0.scatter(self.x, self.y, c='k', s=5, linewidth=0)
        ax0.plot(self.x, self.yp, '-r', linewidth=3)
        ax0.set_title('Original', fontsize=8)

        # plot corrected data
        ax1.scatter(self.x, self.yc, c='k', s=5, linewidth=0)
        ax1.set_title('Corrected', fontsize=8)
