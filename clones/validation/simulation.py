import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ..annotation.bayesian import BayesianClassifier
from ..annotation.genotype import CommunityBasedGenotype
from ..spatial.graphs import WeightedGraph
from .scoring import Scoring

# import warnings
# warnings.filterwarnings('error')


class BenchmarkProperties:
    """ Properties for Benchmark class. """

    @property
    def xy(self):
        """ Measurement spatial coordinates. """
        return self.df[['centroid_x', 'centroid_y']].values

    @property
    def cell_classifier(self):
        return self.annotator.cell_classifier

    @property
    def true_genotypes(self):
        """ True genotypes. """
        return self.df.ground.values

    @property
    def simple_genotypes(self):
        """ Cell-based classifier genotypes. """
        return self.df.simple_genotype.values

    @property
    def community_genotypes(self):
        """ Community-based classifier genotypes. """
        return self.df.community_genotype.values

    @property
    def fluorescence(self):
        """ Measured fluorescence. """
        return self.df[self.annotator.cell_classifier.classify_on].values

    @property
    def simple_MAE(self):
        """ Mean absolute error of cell-based classifier labels. """
        return self.scores['simple'].MAE

    @property
    def community_MAE(self):
        """ Mean absolute error of community-based classifier labels. """
        return self.scores['community'].MAE


class BenchmarkVisualization:
    """ Visualization methods for Benchmark class. """

    @property
    def norm(self):
        """ Diploid normalization. """
        return Normalize(vmin=0, vmax=2)

    @property
    def fnorm(self):
        """ Fluorescence normalization. """
        max_fluorescence = np.percentile(self.cell_classifier.values, 99)
        return Normalize(vmin=0, vmax=max_fluorescence)

    def _scatter(self, ax, c,
                 s=5,
                 cmap=plt.cm.viridis,
                 norm=None,
                 title=None,
                 **kwargs):
        """ Scatter cells in space. """
        if norm is None:
            norm = self.norm
        colors = cmap(norm(c))
        ax.scatter(*self.xy.T, c=colors, s=s, lw=0, **kwargs)
        ax.set_aspect(1)
        ax.axis('off')
        if title is not None:
            ax.set_title(title, fontsize=12)

    def plot_measurements(self, ax=None, norm=None, **kwargs):
        """ Scatter plot of measurement data. """
        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 2))
        c = self.fluorescence

        # define normalization
        if norm is None:
            norm = self.fnorm

        self._scatter(ax, c=c, norm=norm, title=None, **kwargs)

    def show(self, **kwargs):
        """ Plot visual comparison of cell classifiers. """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
        self._scatter(axes[0, 0], c=self.true_genotypes, title='Ground Truth', **kwargs)
        self._scatter(axes[0, 1], c=self.fluorescence, norm=self.fnorm, title='Fluorescence', **kwargs)
        self._scatter(axes[1, 0], c=self.simple_genotypes, title='Cell-based classifier', **kwargs)
        self._scatter(axes[1, 1], c=self.community_genotypes, title='Community-based classifier', **kwargs)

    def show_measurements(self, **kwargs):
        """ Plot visual comparison of genotypes and measurements. """
        fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
        self._scatter(axes[0], c=self.true_genotypes, title='Ground Truth', **kwargs)
        self._scatter(axes[1], c=self.fluorescence, norm=self.fnorm, title='Fluorescence', **kwargs)

    def show_comparison(self, **kwargs):
        """ Plot visual comparison of cell classifiers. """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
        self._scatter(axes[0, 0], c=self.simple_genotypes, title='Cell-based classifier', **kwargs)
        self._scatter(axes[1, 0], c=self.community_genotypes, title='Community-based classifier', **kwargs)
        self.scores['simple'].plot_matrix(ax=axes[0, 1])
        self.scores['community'].plot_matrix(ax=axes[1, 1])


class SimulationBenchmark(BenchmarkProperties, BenchmarkVisualization):
    """
    Class for benchmarking a synthetic simulation.

    Attributes:

        df (pd.DataFrame) - synthetic measurement data

        annotator (CommunityBasedGenotype) - cluster-based annotator

        classify_on (str) - attribute on which cell measurements are classified

    """

    def __init__(self, measurements,
                 classifier=None,
                 classify_on='fluorescence',
                 twolevel=False,
                 rule='weighted'):
        """
        Args:

            measurements (pd.DataFrame) - synthetic measurement data

            classifier (BayesianClassifier) - if None, fit to measurements

            classify_on (str) - attribute on which measurements are classified

            twolevel (bool) - if True, perform two-level clustering

            rule (str) - voting rule, e.g. 'weighted' or 'majority'

        """

        # fit cell classifier and graph
        if classifier is None:
            classifier = self.fit_cell_classifier(measurements, classify_on)

        # assign cell-based labels
        measurements.loc[:, 'simple_genotype'] = classifier(measurements)

        # build graph
        graph = self.build_graph(measurements, classify_on)

        # annotate measurements
        kw = dict(rule=rule, twolevel=twolevel)
        self.annotator = CommunityBasedGenotype(graph, classifier, **kw)
        self.annotator(measurements)
        self.df = measurements

        # score annotation performance
        self.scores = {}
        self.scores['simple'] = self.score(self.simple_genotypes)
        self.scores['community'] = self.score(self.community_genotypes)

    @staticmethod
    def fit_cell_classifier(measurements, classify_on='fluorescence'):
        """ Returns BayesianClassifier object. """
        values = measurements[classify_on].values
        return BayesianClassifier(values, classify_on=classify_on)

    @staticmethod
    def build_graph(measurements, weighted_by='fluorescence'):
        """ Returns WeightedGraph object. """
        return WeightedGraph(measurements, weighted_by=weighted_by)

    def score(self, labels):
        """ Assess accuracy of <labels>. """
        return Scoring(self.true_genotypes, labels)
