import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu

from ..vis.settings import *


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

    def compare(self, test='MW'):
        """
        Run statistical test.

        Args:

            test (str) - name of test used, one of ('KS', 't', 'MW')

        Returns:

            p (float) - p-value

        """

        # extract compared values for each population
        x, y = self.x[self.basis], self.y[self.basis]

        # perform statistical test
        if test == 'KS':
            k, p = ks_2samp(x, y)
        elif test == 't':
            k, p = ttest_ind(x, y)
        else:
            k, p = mannwhitneyu(x, y, alternative='two-sided')

        return p


class CloneComparison(PairwiseComparison):
    """
    Pairwise statistical comparison between two concurrent cell types.

    Attributes:

        type1 (str) - first cell type

        type2 (str) - second cell type

    Inherited attributes:

        x (pd.DataFrame) - measurements of first population

        y (pd.DataFrame) - measurements of second population

        basis (str) - attribute on which populations are compared

    """

    def __init__(self, measurements, type1, type2, basis):
        """
        Instantiate comparison between two concurrent cell types.

        Args:

            measurements (pd.DataFrame) - cell measurement data

            type1 (str) - first cell type

            type2 (str) - second cell type

            basis (str) - attribute on which populations are compared

        """

        # store cell types
        self.type1 = type1
        self.type2 = type2

        # select concurrent cells of each type
        ind = np.logical_and(measurements['concurrent_'+type1],
                             measurements['concurrent_'+type2])
        measurements = measurements[ind]

        # split into two populations
        x = measurements[measurements.celltype == type1]
        y = measurements[measurements.celltype == type2]

        # instantiate comparison
        super().__init__(x, y, basis=basis)

    def plot(self,
             ax=None,
             mode='violin',
             ylabel=None,
             **kwargs):
        """
        Visualize comparison using seaborn box or violinplot.

        Args:

            ax (matplotlib.axes.AxesSubplot)

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
                    x='celltype',
                    y=self.basis,
                    data=pd.concat((self.x, self.y)),
                    order=(self.type1, self.type2),
                    notch=True,
                    width=0.8,
                    **kwargs)

        # plot violinplot
        elif mode == 'violin':
            sns.violinplot(ax=ax,
                    x='celltype',
                    y=self.basis,
                    data=pd.concat((self.x, self.y)),
                    order=(self.type1, self.type2),
                    scale='width',
                    linewidth=0.5,
                    **kwargs)

        # plot stripplot
        elif mode == 'strip':
            sns.stripplot(ax=ax,
                    x='celltype',
                    y=self.basis,
                    data=pd.concat((self.x, self.y)),
                    order=(self.type1, self.type2),
                    dodge=True,
                    **kwargs)

        # format axis
        self.format_axis(ax, ylabel=ylabel, mode=mode)

    def format_axis(self, ax, mode='violin', ylabel=None):
        """
        Format axis.

        Args:

            ax (matplotlib.axes.AxesSubplot)

            mode (str) - type of comparison, either 'box', 'violin', or 'strip'

            ylabel (str) - label for y axis

        """

        # define labels and corresponding fill colors
        labels = dict(m='−/−', h='−/+', w='+/+')
        colors = dict(m='y', h='c', w='m')

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
            color = colors[label.get_text()[0]]
            if mode == 'violin':
                polys[i].set_color(color)
            else:
                ax.artists[i].set_facecolor(color)

            # set tick label as genotype
            label.set_text(labels[label.get_text()[0]])
            ticklabels.append(label)

        # format xlabels
        ax.set_xlabel('')
        _ = ax.set_xticklabels(ticklabels, ha='center')

        # set ylabel
        if ylabel is not None:
            ax.set_ylabel(ylabel)


class SummaryStatistics:
    """
    Summary of comparisons between mutant, heterozygote, and wildtype clones.

    Attributes:

        measurements (pd.DataFrame) - cell measurement data

        basis (str) - attribute on which clones are compared

        test (str) - name of test used, one of ('KS', 't', 'MW')

    """

    def __init__(self, control, perturbation, basis, test='MW'):
        """
        Instantiate summary of comaprisons between mutant, heterozygote, and wildtype clones.

        Args:

            control (pd.DataFrame) - measurements for control condition

            perturbation (pd.DataFrame) - measurements for perturbation condition

            basis (str) - attribute on which clones are compared

            test (str) - name of test used, one of ('KS', 't', 'MW')

        """
        self.measurements = dict(control=control, perturbation=perturbation)
        self.basis = basis
        self.test = test

        # compute and report pvalues
        pvals = self.run()
        self.report(pvals)

    def compare_celltype(self, condition, type1, type2):
        """
        Args:

            condition (str) - experimental condition, 'control' or 'perturbation'

            type1 (str) - first cell type

            type2 (str) - second cell type

        Returns:

            p (float) - p value for comparison statistic

        """
        data = self.measurements[condition]
        comparison = CloneComparison(data, type1, type2, self.basis)
        return comparison.compare(self.test)

    def run(self):
        """
        Compare mutant vs heterozygous and heterozygous vs wildtype for both the control and perturbation conditions.

        Returns:

            pvals (dict) - {comparison: pvalue} pairs

        """
        pvals = dict(
            c_mh = self.compare_celltype('control', 'm', 'h'),
            c_hw = self.compare_celltype('control', 'h', 'w'),
            p_mh = self.compare_celltype('perturbation', 'm', 'h'),
            p_hw = self.compare_celltype('perturbation', 'h', 'w'))
        return pvals

    def report(self, pvals):
        """
        Print summary of statistical comparisons for each condition.

        Args:

            pvals (dict) - {comparison: pvalue} pairs

        """
        print('Statistical test: {}'.format(self.test))
        print('Control: 0x vs 1x: {:0.4f}'.format(pvals['c_mh']))
        print('Control: 1x vs 2x: {:0.4f}'.format(pvals['c_hw']))
        print('Perturbation: 0x vs 1x: {:0.4f}'.format(pvals['p_mh']))
        print('Perturbation: 1x vs 2x: {:0.4f}'.format(pvals['p_hw']))
