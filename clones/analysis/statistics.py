import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu


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

    def plot(self, **kwargs):
        """
        Visualize comparison.

        kwargs: keyword arguments for sns.boxplot
        """
        sns.boxplot(x='celltype',
                    y=self.basis,
                    data=pd.concat((self.x, self.y)),
                    order=(self.type1, self.type2),
                    notch=True,
                    **kwargs)


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
