from os.path import join
import numpy as np
import matplotlib.pyplot as plt


class BenchmarkingResults:
    """
    Container for managing aggregated results of a benchmarking sweep.
    """

    def __init__(self, data, shape, methods=None):
        """
        Instantiate benchmarking results object.

        Args:

            methods (array like) - methods included

        """

        if methods is None:
            methods = ['labels', 'level_only', 'spatial_only']
            methods = [m for m in methods if m in data.columns]
        self.methods = {m: i for i, m in enumerate(methods)}

        columns = ['row_id', 'column_id', 'ambiguity_id']
        self.data = data.groupby(columns)[methods].mean()
        self.shape = shape

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
    def format_ax(ax):
        """ Apply basic formatting to axis. """
        ax.set_ylabel('Fluorescence ambiguity')
        ax.set_xlabel('Clone size')

    @staticmethod
    def plot_image(ax, im, fliplr=False, flipud=False, **kwargs):
        """ Plot <im> on <ax>. """

        if fliplr:
            im = np.fliplr(im)

        if flipud:
            im = np.flipud(im)

        ax.imshow(im, **kwargs)

    def plot_relative_error(self,
                            ax=None,
                            method='labels',
                            reference_method='level_only',
                            row_id=0,
                            vmin=-3,
                            vmax=3,
                            cmap=plt.cm.seismic,
                            **kwargs):
        """
        Plots relative error rate for a given <row_id>.

        """

        # compile foldchange array (FC < 0 is good performance)
        #drop_last_axis = lambda x: x.reshape(x.shape[:-1])
        #matrices = np.split(self.slice(row_id), self.num_methods, -1)
        #matrices = [drop_last_axis(x) for x in matrices]
        #scores = matrices[self.methods[method]]
        #reference_scores = matrices[self.methods[reference_method]]

        scores = self.slice(row_id)[:,:, self.methods[method]]
        reference_scores = self.slice(row_id)[:,:, self.methods[reference_method]]
        foldchange = np.log2(scores/reference_scores)

        # plot
        if ax is None:
            fig, ax = self.build_figure()
        else:
            fig = plt.gcf()

        kw = dict(vmin=vmin, vmax=vmax, cmap=cmap)
        kw.update(kwargs)
        self.plot_image(ax, foldchange, fliplr=True, flipud=False, **kw)
        self.format_ax(ax)

        return fig

    def plot_absolute_error(self,
                            ax=None,
                            method='labels',
                            row_id=0,
                            log=True,
                            vmin=0.,
                            vmax=0.25,
                            cmap=plt.cm.Greys,
                            **kwargs):
        """
        Plots absolute error rates for a given <row_id> and <method>.
        """

        # compile absolute error arrays (low error is good performance)
        # drop_last_axis = lambda x: x.reshape(x.shape[:-1])
        # matrices = np.split(self.slice(row_id), self.num_methods, -1)
        # matrices = [drop_last_axis(x) for x in matrices]
        matrix = self.slice(row_id)[:,:,self.methods[method]]

        # log-transform values
        if log:
            matrix = np.log10(matrix)
            vmin, vmax = np.log10(vmin), np.log10(vmax)

        # plot
        if ax is None:
            fig, axes = self.build_figure(figsize=(2, 2))
        else:
            fig = plt.gcf()

        # plot
        kw = dict(vmin=vmin, vmax=vmax, cmap=cmap)
        kw.update(kwargs)
        self.plot_image(ax, matrix, fliplr=True, flipud=False, **kw)
        self.format_ax(ax)

        return fig
