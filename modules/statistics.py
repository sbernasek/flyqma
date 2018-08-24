import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu


class BoxPlot:

    def __init__(self, df, y='red', hue='cell_type', **kw):
        order = ('control', 'perturbation')
        hue_order = ('m', 'h', 'w')
        self.fig = plt.figure()
        self._plot(df, y, order, hue, hue_order, **kw)
        ax = plt.gca()
        ax.legend(loc=1)
        ax.grid(axis='y')

    @staticmethod
    def _plot(df, y, order, hue, hue_order, **kw):
        sns.boxplot(x='experiment', y=y, data=df,
                    order=order, hue='cell_type', hue_order=hue_order,
                    notch=True, width=0.8, **kw)

class ViolinPlot(BoxPlot):

    @staticmethod
    def _plot(df, y, order, hue, hue_order, **kw):
        sns.violinplot(x='experiment', y=y, data=df,
                    order=order, hue='cell_type', hue_order=hue_order,
                    scale='width', **kw)

class StripPlot(BoxPlot):

    @staticmethod
    def _plot(df, y, order, hue, hue_order, **kw):
        sns.stripplot(x='experiment', y=y, data=df,
                    order=order, hue='cell_type', hue_order=hue_order,
                    dodge=True, **kw)







class PairwiseComparison:

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
    def __init__(self, df, experiment, a, b, basis='green_corrected'):
        # select concurrent cells from specified experiment
        self.a = a
        self.b = b
        cells = df[df.experiment==experiment]
        cells = cells[cells['concurrent_'+a+b]]
        x, y = (cells[cells.cell_type == a]), (cells[cells.cell_type == b])
        PairwiseComparison.__init__(self, x, y, basis=basis)

    def plot(self, ax=None, figsize=(2, 4), **kw):

        # create figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # boxplot
        order = (self.a, self.b)
        data = pd.concat((self.x, self.y))
        sns.boxplot(x='cell_type', y=self.basis, data=data, order=order, notch=True, ax=ax, **kw)
        ax.grid(axis='y')
        return fig



class Summary:
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



