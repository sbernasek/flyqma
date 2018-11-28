import matplotlib.pyplot as plt
from ..bayesian import BayesianClassifier
from ..genotype import CommunityBasedGenotype
from ...spatial.graphs import WeightedGraph
from .scoring import Scoring


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
        return self.df[self.classify_on].values


class BenchmarkVisualization:
    """ Visualization methods for Benchmark class. """

    def _scatter(self, ax, c, s=3, label=''):
        """ Scatter cells in space. """
        ax.scatter(*self.xy.T, c=c, s=s)
        ax.set_aspect(1)
        ax.axis('off')
        ax.set_title(label, fontsize=12)

    def show(self):
        """ Plot visual comparison of cell classifiers. """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
        self._scatter(axes[0, 0], c=self.true_genotypes, label='Ground Truth')
        self._scatter(axes[0, 1], c=self.fluorescence, label='Fluorescence')
        self._scatter(axes[1, 0], c=self.simple_genotypes, label='Cell-based classifier')
        self._scatter(axes[1, 1], c=self.community_genotypes, label='Community-based classifier')

    def show_measurements(self):
        """ Plot visual comparison of genotypes and measurements. """
        fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
        self._scatter(axes[0], c=self.true_genotypes, label='Ground Truth')
        self._scatter(axes[1], c=self.fluorescence, label='Fluorescence')

    def show_comparison(self, **kwargs):
        """ Plot visual comparison of cell classifiers. """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
        self._scatter(axes[0, 0], c=self.simple_genotypes, label='Cell-based classifier')
        self._scatter(axes[1, 0], c=self.community_genotypes, label='Community-based classifier')
        self.scores['simple'].plot_matrix(ax=axes[0, 1], **kwargs)
        self.scores['community'].plot_matrix(ax=axes[1, 1], **kwargs)


class Benchmark(BenchmarkProperties, BenchmarkVisualization):
    """
    Class containing a synthetic benchmark.

    Attributes:

        df (pd.DataFrame) - synthetic measurement data

        annotator (CommunityBasedGenotype) - object for cluster-based annotation

        classify_on (str) - attribute on which cell measurements are classified

    """

    def __init__(self, measurements, classify_on='r', q=100):
        """
        Args:

            measurements (pd.DataFrame) - synthetic measurement data

            classify_on (str) - attribute on which cell measurements are classified

            q (float) - maximum quantile of included edge distances

        """

        self.classify_on = classify_on

        # fit cell classifier and graph
        cell_classifier = self.fit_cell_classifier(measurements, classify_on=classify_on)
        measurements['simple_genotype'] = cell_classifier.labels
        graph = self.build_graph(measurements, classify_on, q=q)

        # annotate measurements
        self.annotator = CommunityBasedGenotype(graph, cell_classifier)
        self.annotator(measurements)
        self.df = measurements

        # score annotation performance
        self.scores = {}
        self.scores['simple'] = self.score(self.simple_genotypes)
        self.scores['community'] = self.score(self.community_genotypes)

    @staticmethod
    def fit_cell_classifier(measurements, classify_on='r'):
        """ Returns BayesianClassifier object. """
        values = measurements[classify_on].values
        return BayesianClassifier(values, classify_on=classify_on)

    @staticmethod
    def build_graph(measurements, weighted_by='r', q=100):
        """ Returns WeightedGraph object. """
        return WeightedGraph(measurements, weighted_by=weighted_by, q=q)

    def score(self, labels):
        """ Assess accuracy of <labels>. """
        return Scoring(self.true_genotypes, labels)
