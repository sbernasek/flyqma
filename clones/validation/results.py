from os.path import join
import numpy as np
import matplotlib.pyplot as plt


class BenchmarkingResults:
    """
    Container for managing aggregated results of a benchmarking sweep.
    """

    def __init__(self, data, shape):
        """
        Instantiate benchmarking results object.
        """

        columns = ['row_id', 'column_id', 'scale_id']
        self.data = data.groupby(columns)[['simple', 'community']].mean()
        self.shape = shape

    def slice(self, row_id=0):
        """
        Returns score matrices for the specified <row_id>. Row IDs correspond to recombination rates.
        """
        df = self.data.xs(row_id, level=0).swaplevel().sort_index(level=0)
        shape = [len(s) for s in df.index.levels]
        matrices = df.values.reshape(*shape, 2)
        simple, community = matrices[:,:,0], matrices[:,:,1]
        return simple, community

    @staticmethod
    def build_figure(nrows=1, ncols=1, figsize=(2, 2)):
        """ Returns figure and axis. """
        return plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    @staticmethod
    def format_ax(ax):
        """ Apply basic formatting to axis. """
        ax.invert_xaxis()
        ax.set_ylabel('Fluorescence ambiguity')
        ax.set_xlabel('Clone size')

    def plot_relative_error(self,
                            row_id=0,
                            vmin=-3,
                            vmax=3,
                            cmap=plt.cm.seismic,
                            **kwargs):
        """ Plots relative error rate for a given <row_id>. """

        # compile foldchange array (FC < 0 is good performance)
        simple, community = self.slice(row_id)
        foldchange = np.log2(community/simple)

        # plot
        fig, ax = self.build_figure()
        ax.imshow(foldchange, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
        self.format_ax(ax)

        return fig

    def plot_absolute_error(self,
                            row_id=0,
                            log=True,
                            vmin=0.,
                            vmax=0.25,
                            cmap=plt.cm.Greys,
                            **kwargs):
        """ Plots absolute error rates for a given <row_id>. """

        # compile absolute error arrays (low error is good performance)
        matrices = self.slice(row_id)

        # plot
        fig, axes = self.build_figure(ncols=2, figsize=(4.5, 2))

        for ax, matrix in zip(axes, matrices):

            # log-transform values
            if log:
                matrix = np.log10(matrix)
                vmin, vmax = -3, -0.5

            # plot
            ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
            self.format_ax(ax)

        return fig
