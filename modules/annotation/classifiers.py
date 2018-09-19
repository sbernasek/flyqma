from os.path import join, exists
from os import mkdir
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib.colors import ListedColormap, Normalize

from ..utilities.io import IO
from ..vis.settings import *


class KM:
    """
    K-means classifier.

    Attributes:
    values (array like) - basis for clustering
    log (bool) - indicates whether clustering performed on log values
    n (int) - number of clusters
    groups (dict) - {cluster_id: group_id} pairs for merging clusters
    cluster_to_groups (vectorized func) - maps cluster_id to group_id
    km (sklearn.cluster.KMeans) - kmeans object
    classifier (vectorized func) - maps value to group_id
    labels (np.ndarray[int]) - group_id labels assigned to fitted values
    cmap (matplotlib.colors.ColorMap) - colormap for group_id
    parameters (dict) - {param name: param value} pairs
    fig (matplotlib.figures.Figure) - histogram figure
    """

    def __init__(self, values, n=3, groups=None, log=False, cmap=None):

        # set values and whether to log transform them
        self.values = values
        self.log = log

        # set groups and number of clusters
        if groups is None:
            groups = {k: k for k in range(n)}
        else:
            groups = {int(k): v for k, v in groups.items()}
            assert n == len(groups), 'Wrong number of groups.'
        self.n = n
        self.groups = groups
        self.cluster_to_group = np.vectorize(groups.get)

        # build classifiers
        x = self.get_values()
        self.km = self._kmeans(x, self.n)
        self.classifier = self._build_value_to_groups_classifier()

        # assign group labels
        self.labels = self.classifier(x.reshape(-1, 1))

        # set colormap
        self.set_cmap(cmap)

        # store parameters
        self.parameters = dict(n=self.n, groups=self.groups, log=self.log)
        self.fig = None

    def __call__(self, x):
        return self.classifier(x)

    @staticmethod
    def _kmeans(x, n):
        """ Fit n clusters to x """
        return KMeans(n).fit(x.reshape(-1, 1))

    @staticmethod
    def _build_value_to_cluster_classifier(km):
        """ Build classifier mapping values to sequential clusters. """
        centroids = km.cluster_centers_.ravel()
        flip = lambda f: f.__class__(map(reversed, f.items()))
        km_to_ordered_dict = flip(dict(enumerate(np.argsort(centroids))))
        km_to_ordered = np.vectorize(km_to_ordered_dict.get)
        classifier = lambda x: km_to_ordered(km.predict(x))
        return classifier

    def _build_value_to_groups_classifier(self):
        """ Build classifier mapping values to groups. """
        value_to_cluster = self._build_value_to_cluster_classifier(self.km)
        classifier = lambda x: self.cluster_to_group(value_to_cluster(x))
        return classifier

    def get_values(self):
        """ Get values for classifier. """
        values = self.values
        if self.log:
            values = np.log10(values)
        return values

    def set_cmap(self, cmap=None):
        """ Set colormap for class labels. """
        if cmap is None:
            self.cmap = ListedColormap(['y', 'c', 'm'], 'indexed', N=3)
        else:
            self.cmap = cmap

    def show(self, **kw):
        """ Plot histogram. """
        self.fig = self._show(self.get_values(), self.labels, self.cmap, **kw)

    @staticmethod
    def _show(x, labels, cmap, ax=None):
        """ Plot histogram. """
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 2))
        else:
            fig = plt.gcf()
        bins = np.linspace(x.min(), x.max(), 50)
        for label in set(labels):
            xi = x[(labels==label)]
            ax.hist(xi, bins=bins, facecolor=cmap(label))

        # format axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Values', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)

        return fig


class CellClassifier(KM):
    """
    K-means based classifier for assigning labels to individual cells.

    Attributes:
    classify_on (str) - cell attribute on which clustering occurs

    Inherited attributes:
    values (array like) - basis for clustering
    log (bool) - indicates whether clustering performed on log values
    n (int) - number of clusters
    groups (dict) - {cluster_id: group_id} pairs for merging clusters
    cluster_to_groups (vectorized func) - maps cluster_id to group_id
    km (sklearn.cluster.KMeans) - kmeans object
    classifier (vectorized func) - maps value to group_id
    labels (np.ndarray[int]) - group_id labels assigned to fitted values
    cmap (matplotlib.colors.ColorMap) - colormap for group_id
    parameters (dict) - {param name: param value} pairs
    fig (matplotlib.figures.Figure) - histogram figure
    """

    def __init__(self, values, classify_on='r_normalized', **kw):
        super().__init__(values, **kw)
        self.classify_on = classify_on
        self.parameters['classify_on'] = classify_on

    def __call__(self, df):
        x =  df[self.classify_on].values.reshape(-1, 1)
        if self.log:
            x = np.log10(x)
        return self.classifier(x)

    @staticmethod
    def from_cells(df, classify_on='r_normalized', **kw):
        values = df[classify_on].values
        return CellClassifier(values, classify_on, **kw)

    @staticmethod
    def from_im_clusters(df, by=None, classify_on='r_normalized', **kw):
        if by is None:
            by = ('disc_genotype', 'disc_id', 'layer', 'im_label')
        values = df.groupby(by=by)[classify_on].mean().values
        return CellClassifier(values, classify_on, **kw)

    def save(self, stack_path, image=True):
        """ Save classifier to specified path. """

        # create directory for classifier
        path = join(stack_path, 'cell_classifier')
        if not exists(path):
            mkdir(path)

        # save values
        np.save(join(path, 'values.npy'), self.values)

        # save parameters
        io = IO()
        io.write_json(join(path, 'parameters.json'), self.parameters)

        # save image
        if image and (self.fig is not None):
            kw = dict(dpi=300, format='pdf', transparent=True, rasterized=True)
            self.fig.savefig(join(path, 'classifier.pdf'), **kw)

    @classmethod
    def load(cls, path):
        """
        Load classifier from file.

        Args:
        path (str) - path to classifier directory

        Returns:
        classifier (CellClassifier)
        """
        io = IO()
        values = io.read_npy(join(path, 'values.npy'))
        parameters = io.read_json(join(path, 'parameters.json'))
        return CellClassifier(values, **parameters)

    @staticmethod
    def _show(x, labels, cmap, ax=None):
        fig = KM._show(x, labels, cmap, ax=ax)
        ax = fig.axes[0]
        ax.set_xlabel('RFP level', fontsize=8)
        ax.set_ylabel('Number of nuclei', fontsize=8)
        return fig


class CommunityClassifier:
    """
    Classifier for assigning labels to communities.

    Attributes:
    classifier (CellClassifier) - individual cell classifier
    """

    def __init__(self, cells, cell_classifier):
        """ Instantiate community classifier. """
        self.classifier = self.build_classifier(cells, cell_classifier)

    def __call__(self, communities):
        """ Classify communities. """
        return self.classifier(communities)

    @staticmethod
    def from_layer(layer, cell_classifier):
        """ Instantiate community classifier from a layer. """
        return CommunityClassifier(layer.df, cell_classifier)

    @staticmethod
    def get_mode(x):
        """ Returns most common value in an array. """
        mode, count = Counter(x).most_common(1)[0]
        return mode

    @classmethod
    def build_classifier(cls, cells, cell_classifier):
        """
        Build classifier assigning genotypes to graph communities.

        Args:
        cells (pd.DataFrame) - cell data including community labels
        cell_classifier (CellClassifier) - assigns genotype to individual cell
        """
        majority_vote = lambda x: cls.get_mode(cell_classifier(x))
        community_to_genotype = cells.groupby('community').apply(majority_vote).to_dict()
        community_to_genotype[-1] = -1
        return np.vectorize(community_to_genotype.get)
