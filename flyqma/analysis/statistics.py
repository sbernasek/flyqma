from itertools import combinations
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu

from ..visualization import *


class PairwiseComparison:
    """
    Pairwise statistical comparison between measurements of two populations.

    Attributes:

        x (pd.DataFrame) - measurements of first population

        y (pd.DataFrame) - measurements of second population

        basis (str) - attribute on which populations are compared

    """

    def __init__(self, x, y, basis):
        """
        Instantiate pairwise comparison of two populations.

        Attributes:

            x (pd.DataFrame) - measurements of first population

            y (pd.DataFrame) - measurements of second population

            basis (str) - attribute on which populations are compared

            """
        self.x = x
        self.y = y
        self.basis = basis

    @property
    def is_greater_than(self):
        """ True if first population mean is greater than the second."""
        return self.x[self.basis].mean() > self.y[self.basis].mean()

    def compare(self, test='MW', **kwargs):
        """
        Run statistical test.

        Args:

            test (str) - name of test used, one of ('KS', 't', 'MW')

        Returns:

            p (float) - p-value

            is_greater_than (bool) - True if first population mean is greater

            kwargs: keyword arguments for statistical test

        """

        # extract compared values for each population
        x, y = self.x[self.basis], self.y[self.basis]

        # perform statistical test
        if test.lower() == 'ks':
            k, p = ks_2samp(x, y, **kwargs)
        elif test.lower() == 't':
            k, p = ttest_ind(x, y, **kwargs)
        elif test.lower() == 'mw':
            k, p = mannwhitneyu(x, y, **kwargs)
        else:
            raise ValueError('Test {:s} not recognized.'.format(test))

        return p, self.is_greater_than


class PairwiseCelltypeComparison(PairwiseComparison):
    """
    Pairwise statistical comparison between two concurrent cell types.

    Attributes:

        label (str) - attribute used to stratify populations

        type1 (str) - first label

        type2 (str) - second label

    Inherited attributes:

        x (pd.DataFrame) - measurements of first population

        y (pd.DataFrame) - measurements of second population

        basis (str) - attribute on which populations are compared

    """

    def __init__(self, measurements, type1, type2, basis,
                 label='celltype',
                 concurrent_only=True):
        """
        Instantiate comparison between two concurrent cell types.

        Args:

            measurements (pd.DataFrame) - measurement data

            type1 (str or int) - first label

            type2 (str or int) - second label

            basis (str) - attribute on which populations are compared

            label (str) - attribute used to define population labels

            concurrent_only (bool) - if True, only compare concurrent cells

        """

        # store labels
        self.label = label
        self.type1 = type1
        self.type2 = type2

        # select concurrent cells of each type
        if concurrent_only:
            k1 = 'concurrent_'+str(self.type1)
            k2 = 'concurrent_'+str(self.type2)
            measurements = measurements[measurements[k1] & measurements[k2]]

        # split into two populations
        x = measurements[measurements[label] == self.type1]
        y = measurements[measurements[label] == self.type2]

        # instantiate comparison
        super().__init__(x, y, basis=basis)

    def plot(self,
             ax=None,
             colors=None,
             mode='violin',
             ylabel=None,
             **kwargs):
        """
        Visualize comparison using seaborn box or violinplot.

        Args:

            ax (matplotlib.axes.AxesSubplot)

            colors (dict) - color for each box/violin keyed by label

            mode (str) - type of comparison, either 'box', 'violin', or 'strip'

            ylabel (str) - label for yaxis

            kwargs: keyword arguments for seaborn plotting function

        """

        # create figure if none provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(1.25, 1.5))

        # plot boxplot
        if mode == 'box':
            sns.boxplot(ax=ax,
                    x=self.label,
                    y=self.basis,
                    data=pd.concat((self.x, self.y)),
                    order=(self.type1, self.type2),
                    notch=True,
                    width=0.8,
                    **kwargs)

        # plot violinplot
        elif mode == 'violin':
            sns.violinplot(ax=ax,
                    x=self.label,
                    y=self.basis,
                    data=pd.concat((self.x, self.y)),
                    order=(self.type1, self.type2),
                    scale='width',
                    linewidth=0.5,
                    **kwargs)

        # plot stripplot
        elif mode == 'strip':
            sns.stripplot(ax=ax,
                    x=self.label,
                    y=self.basis,
                    data=pd.concat((self.x, self.y)),
                    order=(self.type1, self.type2),
                    dodge=True,
                    **kwargs)

        # format axis
        self.format_axis(ax, colors=colors, mode=mode, ylabel=ylabel)

    def format_axis(self, ax,
                    colors=None,
                    axis_labels=None,
                    mode='violin',
                    ylabel=None):
        """
        Format axis.

        Args:

            ax (matplotlib.axes.AxesSubplot)

            colors (dict) - color for each box/violin keyed by label

            axis_labels (dict) - axis label for each box/violin keyed by label

            mode (str) - type of comparison, either 'box', 'violin', or 'strip'

            ylabel (str) - label for y axis

        """

        # format axes
        ax.grid(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        _ = ax.spines['top'].set_visible(False)
        _ = ax.spines['right'].set_visible(False)

        # get violin artists
        if mode == 'violin':
            is_poly = lambda x: x.__class__.__name__ == 'PolyCollection'
            polys = [c for c in ax.collections if is_poly(c)]

        # format xticks
        ticklabels = []

        for i, label in enumerate(ax.get_xticklabels()):

            # set violin/box color
            if colors is not None:
                if mode == 'violin':
                    polys[i].set_color(colors[label.get_text()[0]])
                else:
                    ax.artists[i].set_facecolor(colors[label.get_text()[0]])

            # set tick label as celltype
            if axis_labels is not None:
                label.set_text(axis_labels[label.get_text()[0]])
                ticklabels.append(label)

        # format xlabels
        ax.set_xlabel('')
        if axis_labels is not None:
            _ = ax.set_xticklabels(ticklabels, ha='center')

        # set ylabel
        if ylabel is not None:
            ax.set_ylabel(ylabel)


class CelltypeComparison:
    """
    Summary of comparisons between all labeled celltypes.

    Attributes:

        measurements (pd.DataFrame) - measurement data

        basis (str) - attribute on which populations are compared

        label (str) - attribute used to define population labels

        test (str) - statistical test used, one of ('KS', 't', 'MW')

    """

    def __init__(self, measurements, basis,
                 label='celltype', test='MW', **kwargs):
        """
        Instantiate summary of comparisons between all labeled cell types.

        Args:

            measurements (pd.DataFrame) - measurement data

            basis (str) - attribute on which populations are compared

            label (str) - attribute used to define population labels

            test (str) - name of test used, one of ('KS', 't', 'MW')

            kwargs: keyword arguments for statistical test

        """
        self.measurements = measurements
        self.basis = basis
        self.label = label
        self.test = test

        # compute and report pvalues
        pvals = self.run(**kwargs)
        self.report(pvals)

    @property
    def pairs(self):
        """ Unique pairs of labels. """
        label_values = self.measurements[self.label].unique()
        return list(sorted([sorted(x) for x in combinations(label_values, 2)]))

    def compare_celltype(self, type1, type2, **kwargs):
        """
        Args:

            type1 (str) - first cell type

            type2 (str) - second cell type

            kwargs: keyword arguments for statistical test

        Returns:

            p (float) - p value for comparison statistic

            is_greater_than (bool) - True if first population mean is greater

        """
        comparison = PairwiseCelltypeComparison(self.measurements, type1, type2, self.basis, label=self.label)
        return comparison.compare(self.test, **kwargs)

    def run(self, **kwargs):
        """
        Compare all pairwise combinations of cell types.

        kwargs: keyword arguments for statistical test

        Returns:

            pvals (dict) - {comparison: pvalue} pairs

        """

        signs = {True: ' > ', False: ' < '}

        pvals = OrderedDict()
        for pair in self.pairs:
            pval, greater_than = self.compare_celltype(*pair, **kwargs)
            comparison_name = signs[greater_than].join([str(x) for x in pair])
            pvals[comparison_name] = pval

        return pvals

    def report(self, pvals):
        """
        Print summary of statistical comparisons for each condition.

        Args:

            pvals (dict) - {comparison: pvalue} pairs

        """
        print('Statistical test: {}'.format(self.test))
        for test, pval in pvals.items():
            print(test + ': p = {:0.4f}'.format(pval))


# # define labels and corresponding fill colors
# axis_labels = {'m': '−/−', 'h': '−/+', 'w': '+/+',
#           '0': '−/−', '1': '−/+', '2': '+/+'}

# if colors is None:
#     colors = {'m': '−/−', 'h': '−/+', 'w': '+/+',
#               '0': 'y', '1': 'm', '2': 'c'}
