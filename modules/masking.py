import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from skimage.transform import resize


class FunctionMask:

    def __init__(self, shape, func, res=100):
        self.res = res
        self.shape = shape
        self.values = self.evaluate(func)
        self.vmin, self.vmax = self.values.min(), self.values.max()
        self.apply_threshold(func.threshold)

    def evaluate(self, func):
        yy, xx = np.meshgrid(*(np.arange(0, s, self.res) for s in self.shape))
        grid = np.vstack((xx.ravel(), yy.ravel())).T
        values = resize(func(grid).reshape(yy.shape), self.shape)
        return np.ma.array(values)

    def apply_threshold(self, threshold):
        self.values.mask = (self.values < threshold)

    def add_contour(self, ax, lw=1, color='r'):
        ax.contour(self.values.mask, [0.5], linewidths=[lw], colors=[color])

    def add_contourf(self, ax, alpha=0.5, colors='none'):
        ax.contourf(~self.values.mask, [-.5, .5], colors=['w'], hatches=['//'], alpha=alpha)

    def plot_density(self, mask_alpha=0.5, cmap=plt.cm.plasma, figsize=(5, 5)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.values.data, cmap=cmap, vmin=self.vmin, vmax=self.vmax)
        return ax


class ForegroundMask(FunctionMask):
    pass


class RBFMask:

    def __init__(self, shape, rbf, mask=None, res=100):
        self.rbf = rbf
        self.res = res
        self.shape = shape
        self.zi = self.evaluate(rbf)
        self.label()

        if mask is not None:
            self.apply_mask(mask)

    def evaluate(self, func):
        yy, xx = np.meshgrid(*(np.arange(0, s, self.res) for s in self.shape))
        zi = func(xx, yy)
        return resize(zi, self.shape)

    def label(self):
        labels = np.searchsorted(self.rbf.thresholds, self.zi)
        self.labels = np.ma.array(labels)

    def apply_mask(self, mask):
        self.labels.mask = mask

    def add_contour(self, ax, lw=3, alpha=1, colors=None, cmap=None):
        if cmap is None:
            cmap = plt.cm.Reds

        lmax = self.labels.max()
        levels = np.arange(-0.5, lmax+1, 1)
        if colors is None:
            norm = Normalize(levels.min(), levels.max())
            colors = [cmap(norm(l)) for l in levels]
        ax.contour(self.labels, levels, colors=colors, alpha=alpha, linewidths=lw)

    def add_contourf(self, ax, alpha=0.5, colors=None, cmap=None):

        if cmap is None:
            cmap = plt.cm.Reds

        lmax = self.labels.max()
        levels = np.arange(-0.5, lmax+1, 1)
        if colors is None:
            norm = Normalize(levels.min(), levels.max())
            colors = [cmap(norm(l)) for l in levels]
        ax.contourf(self.labels, levels, colors=colors, alpha=alpha)

    def plot_density(self, cmap=None, figsize=(5, 5)):
        if cmap is None:
            cmap = plt.cm.Reds

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.zi, cmap=cmap)
        ax.axis('off')
        return ax

    def plot_regions(self, s=10, lw=1, cmap=None, figsize=(5, 5)):
        if cmap is None:
            cmap = plt.cm.Reds
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.labels, cmap=cmap, vmin=0, vmax=self.labels.max())
        self.rbf.plot_points(ax, lw=lw, s=s, cmap=cmap)
        ax.axis('off')
        return ax
