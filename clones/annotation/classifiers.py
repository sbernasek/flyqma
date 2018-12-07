from os.path import join, exists
from os import mkdir
import gc
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

from ..utilities.io import IO
from ..vis.settings import *


class Classifier:
    """
    Classifier base class.

    Attributes:

        values (array like) - basis for clustering

        log (bool) - indicates whether clustering performed on log values

        n (int) - number of clusters

        num_labels (int) - number of output labels

        classifier (vectorized func) - maps value to group_id

        labels (np.ndarray[int]) - predicted labels

        cmap (matplotlib.colors.ColorMap) - colormap for group_id

        parameters (dict) - {param name: param value} pairs

        fig (matplotlib.figures.Figure) - histogram figure

    """

    def __init__(self, values, n=3, num_labels=3, log=False, cmap=None):
        """
        Instantiate k-means classifier.

        Args:

            values (array like) - basis for clustering

            n (int) - number of clusters

            num_labels (int) - number of output labels

            log (bool) - indicates whether clustering performed on log values

            cmap (matplotlib.colors.ColorMap) - colormap for cell labels

        """

        # set values, whether to log transform them, and number of clusters
        self._values = values
        self.log = log
        self.n = n
        self.num_labels = num_labels

        # set colormap
        self.set_cmap(cmap=cmap)

        # store parameters
        self.parameters = dict(n=n, num_labels=num_labels, log=log)
        self.fig = None

    def __call__(self, x):
        """ Return class assignments. """
        return self.classifier(x)

    @property
    def values(self):
        """ Values for classifier. """
        if self.log:
            return np.log10(self._values)
        else:
            return self._values

    def set_cmap(self, cmap=None):
        """
        Set colormap for class labels.

        Args:

            cmap (matplotlib.colormap)

        """

        # select colormap
        if cmap is None:
            cmap = plt.cm.viridis

        # normalize
        norm = Normalize(vmin=0, vmax=self.num_labels-1)
        self.cmap = lambda x: cmap(norm(x))

    def show(self, **kw):
        """ Plot histogram. """
        self.fig = self._show(self.values, self.labels, self.cmap, **kw)

    @staticmethod
    def _show(x, labels, cmap,
              bins=None,
              xlim=None,
              histtype='step',
              stacked=False,
              fill=True,
              ax=None,
              **kw):
        """ Plot histogram. """
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 2))
        else:
            fig = plt.gcf()

        if xlim is None:
            xlim = (x.min(), x.max())

        if bins is None:
            bins = np.linspace(*xlim, 25)
        values = [x[(labels==label)] for label in set(labels)]
        colors = [cmap(label) for label in set(labels)]
        ax.hist(values,
                bins=bins,
                color=colors,
                histtype=histtype,
                stacked=stacked,
                fill=fill,
                **kw)

        # format axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Values', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)

        return fig


class KM(Classifier):
    """
    K-means classifier.

    Attributes:

        groups (dict) - {cluster_id: group_id} pairs for merging clusters

        cluster_to_groups (vectorized func) - maps cluster_id to group_id

        km (sklearn.cluster.KMeans) - kmeans object

    Inherited attributes:

        values (array like) - basis for clustering

        log (bool) - indicates whether clustering performed on log values

        n (int) - number of clusters

        classifier (vectorized func) - maps value to group_id

        labels (np.ndarray[int]) - predicted labels

        cmap (matplotlib.colors.ColorMap) - colormap for group_id

        parameters (dict) - {param name: param value} pairs

        fig (matplotlib.figures.Figure) - histogram figure

    """

    def __init__(self, values, n=3, groups=None, log=False, cmap=None):
        """
        Instantiate k-means classifier.

        Args:

            values (array like) - basis for clustering

            n (int) - number of clusters

            groups (dict) - {cluster_id: group_id} pairs for merging clusters

            log (bool) - indicates whether clustering performed on log values

            cmap (matplotlib.colors.ColorMap) - colormap for group_id

        """

        # set groups and number of clusters
        if groups is None:
            groups = {k: k for k in range(n)}
        else:
            groups = {int(k): v for k, v in groups.items()}
            n = len(groups)

        # instantiate classifier
        super().__init__(values, n=n, num_labels=len(groups), log=log, cmap=cmap)
        self.groups = groups
        self.cluster_to_group = np.vectorize(groups.get)

        # build classifiers
        self.km = self._kmeans(self.values, self.n)
        self.classifier = self._build_value_to_groups_classifier()

        # assign group labels
        self.labels = self.classifier(self.values.reshape(-1, 1))

        # store parameters
        self.parameters.update(dict(groups=self.groups))

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

    def __init__(self, values, classify_on='r_normalized', **kwargs):
        """
        Fit a cell classifier to an array of values.

        Args:

            values (np.ndarray[float]) - 1-D vector of measured values

            classify_on (str) - cell measurement attribute from which values came

            kwargs: keyword arguments for k-means classifier

        Returns:

            classifier (CellClassifier)

        """
        super().__init__(values, **kwargs)
        self.classify_on = classify_on
        self.parameters['classify_on'] = classify_on

    def __call__(self, df):
        """ Assign class labels to measurement data. """
        x =  df[self.classify_on].values.reshape(-1, 1)
        if self.log:
            x = np.log10(x)
        return self.classifier(x)

    @staticmethod
    def from_measurements(measurements, classify_on='r_normalized', **kwargs):
        """
        Fit a cell classifier to measurement data.

        Args:

            measurements (pd.DataFrame) - cell measurement data

            classify_on (str) - cell measurement attribute on which to cluster

            kwargs: keyword arguments for k-means classifier

        Returns:

            classifier (CellClassifier)

        """
        values = measurements[classify_on].values
        return CellClassifier(values, classify_on, **kwargs)

    @staticmethod
    def from_im_clusters(df,
                         by=None,
                         classify_on='r_normalized',
                         **kwargs):
        if by is None:
            by = ('disc_genotype', 'disc_id', 'layer', 'im_label')
        values = df.groupby(by=by)[classify_on].mean().values
        return CellClassifier(values, classify_on, **kwargs)

    def save(self, dirpath, image=True):
        """
        Save classifier to specified path.

        Args:

            dirpath (str) - directory in which classifier is to be saved

            image (bool) - if True, save labeled histogram image

        """

        # create directory for classifier
        path = join(dirpath, 'cell_classifier')
        if not exists(path):
            mkdir(path)

        # save values
        np.save(join(path, 'values.npy'), self._values)

        # save parameters
        io = IO()
        io.write_json(join(path, 'parameters.json'), self.parameters)

        # save image
        if image:

            # plot histogram
            self.show()

            # save image
            kw = dict(dpi=100, format='pdf', transparent=True, rasterized=True)
            self.fig.savefig(join(path, 'classifier.pdf'), **kw)
            self.fig.clf()
            plt.close(self.fig)
            gc.collect()

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
