import numpy as np
from ..visualization import *


class GrowthTrends:
    """ Class for extracting and plotting clone trends. """

    def __init__(self, data):
        """
        Instantiate object for plotting clone trend data.

        Args:

            data (pd.DataFrame)

        """
        self.data = data.replace([np.inf, -np.inf], np.nan)
        self.x = 2**self.means.recombination_start
        self.ylabels = {
            'num_clones': 'Number of clones',
            'mean_clone_size': 'Mean clone size',
            'percent_heterozygous': 'Recombination extent',
            'transclone_edges': 'Trans-clone edges per cell',
            'clone_size_variation': 'Clone size variation'}

    @property
    def num_replicates(self):
        """ Number of growth replicates. """
        return self.data.replicate_id.max()+1

    @property
    def means(self):
        """ Mean values aggregated over replicates. """
        gb = self.data.groupby('column_id')
        return gb.mean()

    @property
    def stds(self):
        """ Std dev of values aggregated over replicates. """
        gb = self.data.groupby('column_id')
        return gb.std()

    @default_figure
    def plot_trend(self, yvar, ax=None, **kwargs):
        """ Plot <yvar> against recombination start generation. """

        # get y data
        y = self.means[yvar]
        ystd = self.stds[yvar]
        y_SEM = ystd / np.sqrt(self.num_replicates)

        # plot trend
        ax.errorbar(self.x, y, yerr=y_SEM, **kwargs)

        # format axis
        ax.set_xlabel('Recombination start (no. cells)')
        ylabel = ''
        if yvar in self.ylabels.keys():
            ylabel = self.ylabels[yvar]
        ax.set_ylabel(ylabel)
        ax.set_xscale('log', basex=2)
        ax.set_xticks([1,2,4,8,16,32,64,128,256,512])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
