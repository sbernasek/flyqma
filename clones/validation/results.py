from os.path import join
import numpy as np
import matplotlib.pyplot as plt

from ..visualization import *


class BenchmarkingResults:
    """
    Container for managing aggregated results of a benchmarking sweep.
    """

    def __init__(self, data, shape,
                 methods=None,
                 clone_sizes=None,
                 ambiguities=None):
        """
        Instantiate benchmarking results object.

        Args:

            data (pd.DataFrame) - benchmarking performance data

            shape (tuple) - shape of benchmarking sweep

            methods (array like) - methods included

            clone_sizes (array like) - mean size of clones (x-values)

            ambiguities (array like) - fluorescence ambiguities (y-values)

        """

        if methods is None:
            methods = ['labels_MAE',
                'level_only_MAE',
                'spatial_only_MAE',
                'community_MAE',
                'labels_PCT',
                'level_only_PCT',
                'spatial_only_PCT',
                'community_PCT']
            methods = [m for m in methods if m in data.columns]

        self.methods = {m: i for i, m in enumerate(methods)}

        columns = ['row_id', 'column_id', 'ambiguity_id']
        self.data = data.groupby(columns)[methods].mean()
        self.shape = shape

        self.clone_sizes = clone_sizes
        self.ambiguities = ambiguities

    @property
    def num_methods(self):
        """ Number of methods. """
        return len(self.methods)

    def slice(self, row_id=0):
        """
        Returns score matrices for the specified <row_id>. Row IDs correspond to recombination rates.
        """
        data = self.data.xs(row_id, level=0).swaplevel().sort_index(level=0)
        shape = [len(s) for s in data.index.levels]
        matrices = data.values.reshape(*shape, self.num_methods)
        return matrices

    @staticmethod
    def build_figure(nrows=1, ncols=1, figsize=(2, 2)):
        """ Returns figure and axis. """
        return plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    @staticmethod
    def plot_image(ax, im, fliplr=False, flipud=False, **kwargs):
        """ Plot <im> on <ax>. """

        if fliplr:
            im = np.fliplr(im)

        if flipud:
            im = np.flipud(im)

        ax.imshow(im, **kwargs)

    @default_figure
    def plot_relative_error(self,
                            method='community_MAE',
                            reference_method='level_only_MAE',
                            row_id=0,
                            vmin=-3,
                            vmax=3,
                            cmap=plt.cm.seismic,
                            ax=None,
                            figsize=(2., 2.),
                            **kwargs):
        """
        Plots relative error rate for a given <row_id>.

        """

        scores = self.slice(row_id)[:,:, self.methods[method]]
        reference_scores = self.slice(row_id)[:,:, self.methods[reference_method]]
        foldchange = np.log2(scores/reference_scores)

        kw = dict(vmin=vmin, vmax=vmax, cmap=cmap)
        kw.update(kwargs)
        self.plot_image(ax, foldchange, fliplr=True, flipud=False, **kw)
        self.format_ax(ax)

    @default_figure
    def plot_absolute_error(self,
                            method='community_MAE',
                            row_id=0,
                            log=True,
                            vmin=0.,
                            vmax=0.33,
                            cmap=plt.cm.inferno,
                            ax=None,
                            figsize=(2., 2.),
                            **kwargs):
        """
        Plots absolute error rates for a given <row_id> and <method>.
        """

        matrix = self.slice(row_id)[:,:,self.methods[method]]

        # log-transform values
        if log:
            matrix = np.log10(matrix)
            vmin, vmax = np.log10(vmin), np.log10(vmax)

        # plot
        kw = dict(vmin=vmin, vmax=vmax, cmap=cmap)
        kw.update(kwargs)
        self.plot_image(ax, matrix, fliplr=True, flipud=False, **kw)
        self.format_ax(ax)

    def format_ax(self, ax):
        """ Apply formatting to axis. """

        if self.clone_sizes is not None:
            xvals = self.clone_sizes[::-1]
        else:
            xvals = ax.get_xticks()

        if self.ambiguities is not None:
            yvals = self.ambiguities
        else:
            yvals = ax.get_yticks()

        # set axis extent
        extent = [xvals.min(), xvals.max(), yvals.max(), yvals.min()]
        ax.images[0].set_extent(extent)
        ax.invert_yaxis()
        ax.set_aspect(xvals.ptp()/yvals.ptp())

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Fluorescence ambiguity')
        ax.set_xlabel('Clone size')
