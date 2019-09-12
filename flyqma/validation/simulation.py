from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .training import Training
from .scoring import Scoring

# import warnings
# warnings.filterwarnings('error')


class BenchmarkProperties:
    """ Properties for Benchmark class. """

    @property
    def xykey(self):
        """ Attribute key for spatial coordinates. """
        return ['x', 'y']

    @property
    def xy(self):
        """ Measurement spatial coordinates. """
        return self.data[self.xykey].values

    @property
    def classifier(self):
        return self.annotator.classifier

    @property
    def fluorescence(self):
        """ Measured fluorescence. """
        return self.data[self.attribute].values

    @property
    def ground_truth(self):
        """ True labels. """
        return self.data.true_dosage.values

    @property
    def labels(self):
        """ Labels assigned by bivariate classifier. """
        return self.data.labels.values

    @property
    def level_only(self):
        """ Labels assigned by univariate fluorescence classifier. """
        return self.data.level_only.values

    @property
    def spatial_only(self):
        """ Labels assigned by univariate spatial classifier. """
        return self.data.spatial_only.values

    @property
    def MAE(self):
        """ Mean absolute error of labels. """
        return self.scores['labels'].MAE

    @property
    def MAE_comm(self):
        """ Mean absolute error of labels when classifier is applied to communities. """
        return self.scores['labels_comm'].MAE

    @property
    def MAE_levels(self):
        """ Mean absolute error of level-only labels. """
        return self.scores['level_only'].MAE

    @property
    def MAE_spatial(self):
        """ Mean absolute error of space-only labels. """
        return self.scores['spatial_only'].MAE


class BenchmarkVisualization:
    """ Visualization methods for Benchmark class. """

    @property
    def norm(self):
        """ Diploid normalization. """
        return Normalize(vmin=0, vmax=2)

    @property
    def fnorm(self):
        """ Fluorescence normalization. """
        return Normalize(*self.annotator.classifier.model.bounds)

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
        self._scatter(axes[0, 0], c=self.ground_truth, title='Ground Truth', **kwargs)
        self._scatter(axes[0, 1], c=self.fluorescence, norm=self.fnorm, title='Fluorescence', **kwargs)
        self._scatter(axes[1, 0], c=self.level_only, title='Level only', **kwargs)
        self._scatter(axes[1, 1], c=self.labels, title='Assigned labels', **kwargs)

    def show_classifiers(self, **kwargs):
        """ Plot visual comparison of cell classifiers. """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
        self._scatter(axes[0, 0], c=self.ground_truth, title='Ground Truth', **kwargs)
        self._scatter(axes[0, 1], c=self.labels, title='Assigned labels', **kwargs)
        self._scatter(axes[1, 0], c=self.level_only, title='Level only', **kwargs)
        self._scatter(axes[1, 1], c=self.spatial_only, title='Spatial only', **kwargs)

    def show_measurements(self, **kwargs):
        """ Plot visual comparison of genotypes and measurements. """
        fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
        self._scatter(axes[0], c=self.ground_truth, title='Ground Truth', **kwargs)
        self._scatter(axes[1], c=self.fluorescence, norm=self.fnorm, title='Fluorescence', **kwargs)

    def show_comparison(self, **kwargs):
        """ Plot visual comparison of cell classifiers. """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
        self._scatter(axes[0, 0], c=self.level_only, title='Level only', **kwargs)
        self._scatter(axes[1, 0], c=self.labels, title='Assigned labels', **kwargs)
        self.scores['level_only'].plot_matrix(ax=axes[0, 1])
        self.scores['labels'].plot_matrix(ax=axes[1, 1])


class SimulationBenchmark(Training,
                          BenchmarkProperties,
                          BenchmarkVisualization):
    """
    Class for benchmarking a synthetic simulation.

    Attributes:

        data (pd.DataFrame) - synthetic measurement data

        attribute (str) - attribute on which cell measurements are classified

        annotator (Annotation) - object that assigns labels to measurements

    """

    def __init__(self, measurements,
                 annotator=None,
                 graph=None,
                 attribute='clonal_marker',
                 logratio=True,
                 training_kw={},
                 testing_kw={}):
        """
        Args:

            measurements (pd.DataFrame) - synthetic measurement data

            annotator (Annotation) - if None, fit annotator to measurements

            graph (WeightedGraph) - if None, create a new graph

            attribute (str) - attribute on which measurements are classified

            logratio (bool) - if True, weight graph edges by log-ratio of attribute level. otherwise, use the absolute difference

            training_kw (dict) - keyword arguments for annotator training

            testing_kw (dict) - keyword arguments for annotator application

        """

        self.attribute = attribute

        # build graph
        if graph is None:
            graph = self.build_graph(measurements, attribute, logratio)

        # train annotator
        if annotator is None:
            annotator = self.train(graph, attribute=attribute, **training_kw)
        self.annotator = annotator

        # apply graph-based annotation
        measurements['labels'] = annotator.annotate(graph, **testing_kw)

        # apply graph-based annotation based on community
        community_kw = deepcopy(testing_kw)
        community_kw['sampler_type'] = 'community'
        community_kw['sampler_kwargs'] = {'depth': 1.}
        measurements['labels_comm'] = annotator.annotate(graph, **community_kw)

        # apply univariate annotation using only fluorescence levels
        measurements['level_only'] = annotator.classifier[0](measurements)

        # apply bivariate annotation using only spatial context
        bv_kwargs = deepcopy(testing_kw)
        bv_kwargs['bivariate_only'] = True
        measurements['spatial_only'] = annotator.annotate(graph, **bv_kwargs)

        # store measurements
        self.data = measurements

        # score annotation performance
        self.scores = {}
        self.scores['labels'] = self.score(self.labels)
        self.scores['labels_comm'] = self.score(self.data.labels_comm.values)
        self.scores['level_only'] = self.score(self.level_only)
        self.scores['spatial_only'] = self.score(self.spatial_only)

    def score(self, labels):
        """ Assess accuracy of <labels>. """
        return Scoring(self.ground_truth, labels)
