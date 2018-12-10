import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ..annotation.bayesian import BayesianClassifier
from ..annotation.model_selection import ModelSelection
from ..annotation.community import InfomapLabeler, KatzLabeler
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
    def katz_genotypes(self):
        """ Genotype based on Katz centrality. """
        return self.df.katz_genotype.values

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

    @property
    def katz_MAE(self):
        """ Mean absolute error of Katz classifier labels. """
        return self.scores['katz'].MAE


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

    def show_classifiers(self, **kwargs):
        """ Plot visual comparison of cell classifiers. """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
        self._scatter(axes[0, 0], c=self.true_genotypes, title='Ground Truth', **kwargs)
        self._scatter(axes[0, 1], c=self.simple_genotypes, title='Cell-based classifier', **kwargs)
        self._scatter(axes[1, 0], c=self.community_genotypes, title='Infomap classifier', **kwargs)
        self._scatter(axes[1, 1], c=self.katz_genotypes, title='Katz centrality classifier', **kwargs)

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

        annotator (InfomapLabeler) - cluster-based annotator

        classify_on (str) - attribute on which cell measurements are classified

    """

    def __init__(self, measurements,
                 classifier=None,
                 classify_on='fluorescence',
                 logratio=False,
                 twolevel=False,
                 rule='proportional',
                 katz_kwargs={}):
        """
        Args:

            measurements (pd.DataFrame) - synthetic measurement data

            classifier (BayesianClassifier) - if None, fit to measurements

            classify_on (str) - attribute on which measurements are classified

            logratio (bool) - if True, weight edges by logratio

            twolevel (bool) - if True, perform two-level clustering

            rule (str) - voting rule, e.g. 'proportional', 'weighted' or 'majority'

            katz_kwargs (dict) - keyword arguments for KatzClassifier

        """

        # fit cell classifier and graph
        if classifier is None:
            classifier = self.fit_cell_classifier(measurements, classify_on)

        # assign cell-based labels
        measurements.loc[:, 'simple_genotype'] = classifier(measurements)

        # build graph
        graph = self.build_graph(measurements, classify_on, logratio)

        # annotate measurements using infomap cluster-based labeler
        kw = dict(rule=rule, twolevel=twolevel)
        self.annotator = InfomapLabeler(graph, classifier, **kw)
        self.annotator(measurements)

        # annotate measurements using Katz centrality-based labeler
        katz_annotator = KatzLabeler(graph, classifier, **katz_kwargs)
        katz_annotator(measurements)

        # store measurements
        self.df = measurements

        # score annotation performance
        self.scores = {}
        self.scores['simple'] = self.score(self.simple_genotypes)
        self.scores['community'] = self.score(self.community_genotypes)
        self.scores['katz'] = self.score(self.katz_genotypes)

    @staticmethod
    def fit_cell_classifier(measurements, classify_on='fluorescence'):
        """ Returns BayesianClassifier object. """
        values = measurements[classify_on].values
        selector = ModelSelection(values, classify_on, max_num_components=6)
        return selector.BIC_optimal

    @staticmethod
    def build_graph(measurements, weighted_by='fluorescence', logratio=False):
        """ Returns WeightedGraph object. """
        return WeightedGraph(measurements, weighted_by=weighted_by, logratio=logratio)

    def score(self, labels):
        """ Assess accuracy of <labels>. """
        return Scoring(self.true_genotypes, labels)
