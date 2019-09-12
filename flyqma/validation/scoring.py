import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ..visualization import *
from ..visualization.figures import Figure


class Scoring:
    """
    Object for scoring classification accurancy.

    Attributes:

        data (pd.DataFrame) - data with 'measured' and 'predicted' attributes

        n (int) - number of classifications

        percent_correct (float) - frequency of correct classification

        matrix (ErrorMatrix) - classification adjacency matrix

    """

    def __init__(self, measured, predicted):
        """
        Instantiate scoring prediction.

        Args:

            measured (array like) - manually assigned class labels

            predicted (array like) - predicted class labels

        """
        columns = ('measured', 'predicted')

        if type(measured) in (np.ndarray, list, tuple):
            data = np.vstack((measured, predicted)).T
            data = pd.DataFrame(data, columns=columns)
        elif type(measured) == pd.Series:
            data = pd.DataFrame([measured, predicted], index=columns).T
        else:
            raise ValueError('Type of <measured> not recognized.')

        self.data = data
        self.n = len(self.data)
        self.compare()

    def __add__(self, x):
        """ Combine additively with another scoring matrix. """
        merged = pd.concat([self.data, x.data])
        return Scoring(merged[['measured', 'predicted']])

    @property
    def error(self):
        """ Normalized total classification error. """
        return self.data.difference.sum() / (2*self.n)

    @property
    def percent_correct(self):
        """ Fraction of measurements with correct annotation. """
        return self.data.correct.sum()/self.n

    @property
    def f1(self):
        """ F1 score. """
        values = self.data[['measured', 'predicted']].values
        return f1_score(*values.T, average='weighted')

    @property
    def MAE(self):
        """ Mean absolute error averaged to correct for imbalance. """
        return self.data.groupby('measured')['difference'].mean().mean() / 2

    @property
    def percent_correct_per_layer(self):
        """ Percent correct per layer. """
        return (self.data.correct).groupby(['stack', 'layer']).mean()

    @property
    def MAE_per_layer(self):
        """ Mean absolute error per layer. """
        MAE = lambda x: x.groupby('measured')['difference'].mean().mean() / 2
        return self.data.groupby(['stack', 'layer']).apply(MAE)

    def compare(self):
        """ Evaluate frequency of correct classification. """
        self.data['difference'] = abs(self.data.measured-self.data.predicted)
        self.data['correct'] = (self.data.difference==0)

    def plot_matrix(self, **kwargs):
        """
        Plot classification adjacency matrix.

        kwargs: keyword arguments for adjacency matrix
        """
        measured = self.data.measured.values.astype(int)
        predicted = self.data.predicted.values.astype(int)
        self.matrix = AdjacencyMatrix(measured, predicted, **kwargs)

    @default_figure
    def plot_label_distribution(self,
                                mcolor='tomato',
                                pcolor='darkmagenta',
                                width=0.4,
                                legend=False,
                                ax=None):
        """
        Plot distribution of labels among annotated cells.

        Args:

            mcolor (str) - color for measured labels

            pcolor (str) - color for predicted labels

            width (float) - bar width

            legend (bool) - if True, include legend

        """

        # group data by label-type and obtain counts
        m = self.data.measured.groupby(self.data.measured).count().values
        p = self.data.predicted.groupby(self.data.predicted).count().values

        # plot bars
        ind = np.arange(m.size)
        _ = ax.bar(ind-width/2, m, width, color=mcolor, label='Measured')
        _ = ax.bar(ind+width/2, p, width, color=pcolor, label='Predicted')

        if legend:
            ax.legend(loc=2, frameon=False, fontsize=6,
                       bbox_to_anchor=(0.05, 1.1, 1., .102),
                       ncol=1, borderaxespad=0.)

        # format axis
        ax.set_xticks(ind)
        ax.set_xticklabels(['{:d}x'.format(i) for i in ind])
        ax.set_ylabel('Number of labels')


class AdjacencyMatrix(Figure):
    """
    Classification adjacency matrix. Generates a figure showing the frequency of overlap between predicted and manually assigned labels.

    Attributes:

        counts (np.ndarray[int]) - pairwise overlap between manually assigned and predicted class labels

        fig (matplotlib.figures.Figure) - adjacency matrix figure

    """

    def __init__(self, measured, predicted,
                 ax=None,
                 text=None,
                 figsize=(1.5, 1.5),
                 **kwargs):
        """
        Instantiate error matrix.

        Args:

            measured (array like) - measured class labels

            predicted (array like) - predicted class labels

            ax (matplotlib.axes.AxesSubplot) - if None, create figure

            text (str) - indicates whether 'counts' or 'rates' labels are added

            figsize (tuple) - figure dimensions

            kwargs: keyword arguments for adjacency matrix construction

        """

        self.counts = self.build_matrix(measured, predicted)

        # create figue and add single axis
        if ax is None:
            self.fig = self.create_figure(figsize=figsize)
            self.add_axes()

        else:
            self.fig = plt.gcf()
            self.axes = ax

        # plot classifier adjacency matrix
        self.plot_matrix(self.axes, text=text, **kwargs)

    def save(self, name, dirpath='./', **kwargs):
        """
        Save figure to file.

        Args:

            name (str) - file name without format extension

            dirpath (str) - directory in which to save file

        Keyword arguments:

            fmt (str) - file format, eg 'pdf'

            dpi (int) - resolution

            transparent (bool) - if True, remove background

            rasterized (bool) - if True, rasterize figure data

            kwargs: keyword arguments for plt.savefig

        """
        self._save(self.fig, name, dirpath, **kwargs)

    @staticmethod
    def build_matrix(measured, predicted):
        """
        Compute 2D histogram of measured versus predicted values.

        Args:

            measured (array like) - measured class labels

            predicted (array like) - predicted class labels

        Returns:

            counts (np.ndarray[int]) - pairwise overlap between measured/predicted classes

        """
        counts, _, _ = np.histogram2d(measured, predicted, bins=np.arange(3.5))
        return counts.astype(np.int64)

    def plot_matrix(self, ax, text=None, fontsize=7):
        """
        Plot classification adjacency matrix.

        Args:

            ax (matplotlib.axes.AxesSubplot)

            text (str) - indicates whether 'counts' or 'rates' labels are added

            fontsize (int) - font size for text

        """

        # compute error rates
        rates = self.counts.astype(np.float64)
        rates /= self.counts.sum(axis=1).reshape(-1, 1)

        # plot image
        ax.imshow(rates.T, cmap=plt.cm.Reds, vmin=0, vmax=1)
        ax.invert_yaxis()

        # add labels
        kw = dict(horizontalalignment='center',
                  verticalalignment='center',
                  fontsize=fontsize)

        for i in range(3):
            for j in range(3):

                if i == j:
                    color = 'w'
                    weight = 'bold'
                else:
                    color = 'k'
                    weight ='normal'

                if text=='rates':
                    s = '{:0.0%}'.format(rates[i, j])
                    ax.text(i, j, s=s, color=color, weight=weight, **kw)
                elif text=='counts':
                    s = '{:d}'.format(self.counts[i, j])
                    ax.text(i, j, s=s, color=color, weight=weight, **kw)
                else:
                    continue

        # format axes
        self.format_axis(ax)

    @staticmethod
    def format_axis(ax, labelsize=8):
        """ Format individual axis. """
        ax.set_xlabel('True label', fontsize=labelsize)
        ax.set_ylabel('Assigned label', fontsize=labelsize)
        ax.set_xticks(np.arange(2.5))
        ax.set_xticklabels(['0x', '1x', '2x'], fontsize=labelsize-1)
        ax.set_yticks(np.arange(2.5))
        ax.set_yticklabels(['0x', '1x', '2x'], fontsize=labelsize-1)
        ax.set_aspect(1)
