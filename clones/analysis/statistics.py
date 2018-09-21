import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu


class PairwiseComparison:
    """ Pairwise statistical comparison between two populations. """

    def __init__(self, x, y, basis='green_corrected'):
        self.x = x
        self.y = y
        self.basis = basis

    def compare(self, test='MW'):

        x, y = self.x[self.basis], self.y[self.basis]

        if test == 'KS':
            k, p = ks_2samp(x, y)
        elif test == 't':
            k, p = ttest_ind(x, y)
        else:
            k, p = mannwhitneyu(x, y, alternative='two-sided')

        return p


class CloneComparison(PairwiseComparison):
    """ Pairwise statistical comparison between two clones. """

    def __init__(self, df, experiment, a, b, basis='green_corrected'):
        # select concurrent cells from specified experiment
        self.a = a
        self.b = b
        cells = df[df.experiment==experiment]
        cells = cells[cells['concurrent_'+a+b]]
        x, y = (cells[cells.celltype == a]), (cells[cells.celltype == b])
        PairwiseComparison.__init__(self, x, y, basis=basis)

    def plot(self, ax=None, figsize=(2, 4), **kw):

        # create figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # boxplot
        order = (self.a, self.b)
        data = pd.concat((self.x, self.y))
        sns.boxplot(x='celltype', y=self.basis, data=data, order=order, notch=True, ax=ax, **kw)
        ax.grid(axis='y')
        return fig


class SummaryStatistics:
    """ Summary of statistical comparisons between clone genotypes. """

    def __init__(self, df, basis='green_corrected', test='MW'):
        self.df = df
        self.basis = basis
        self.test = test
        self.pvals = self.compute_pvalues()
        self.report()

    def compare(self, experiment, a, b):
        comparison = CloneComparison(self.df, experiment, a, b, self.basis)
        p = comparison.compare(self.test)
        return p

    def compute_pvalues(self):
        pvals = dict(
            control_mh = self.compare('control', 'm', 'h'),
            control_hw = self.compare('control', 'h', 'w'),
            mutant_mh = self.compare('perturbation', 'm', 'h'),
            mutant_hw = self.compare('perturbation', 'h', 'w'))
        return pvals

    def report(self):
        print('TEST: {}'.format(self.test))
        print('Control: 0x vs 1x: {:0.4f}'.format(self.pvals['control_mh']))
        print('Control: 1x vs 2x: {:0.4f}'.format(self.pvals['control_hw']))
        print('Mutant: 0x vs 1x: {:0.4f}'.format(self.pvals['mutant_mh']))
        print('Mutant: 1x vs 2x: {:0.4f}'.format(self.pvals['mutant_hw']))



