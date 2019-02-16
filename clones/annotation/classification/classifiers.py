from os.path import join, exists
from os import mkdir
import gc
import numpy as np
from sklearn.cluster import k_means
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ...utilities.io import IO


class ClassifierProperties:
    """
    Properties for classifier objects.
    """

    @property
    def num_samples(self):
        """ Number of samples. """
        return len(self._values)

    @property
    def values(self):
        """ Values for classifier. """
        if self.log:
            return np.log(self._values)
        else:
            return self._values

    @property
    def order(self):
        """ Ordered component indices (low to high). """
        x = self.component_to_label
        return sorted(x, key=x.__getitem__)

    @property
    def component_groups(self):
        """ List of lists of components for each label. """
        x = self.component_to_label
        labels = np.unique(list(x.values()))
        return [[k for k, v in x.items() if v == l] for l in labels]

    @property
    def centroids(self):
        """ Means of each component on linear scale (not log transformed). """
        centroids = self.means
        if self.log:
            centroids = np.exp(centroids)
        return centroids.reshape(-1, 1)

    @property
    def component_to_label(self):
        """
        Returns dictionary mapping components to labels.  Mapping is achieved by k-means clustering the model centroids (linear scale).
        """
        n = self.num_labels

        cluster_means, cluster_labels, _ = k_means(self.centroids, n)
        component_to_label = {}
        for label, c in enumerate(np.argsort(cluster_means.ravel())):
            for d in (cluster_labels==c).nonzero()[0]:
                component_to_label[d] = label
        return component_to_label


class Classifier(ClassifierProperties):
    """
    Classifier base class. Children of this class must possess a means attribute, as well as a predict method.


    Attributes:

        values (array like) - basis for clustering

        classify_on (str or list) - attribute(s) on which to cluster

        log (bool) - indicates whether clustering performed on log values

        num_labels (int) - number of output labels

        classifier (vectorized func) - maps value to label_id

        labels (np.ndarray[int]) - predicted labels

        cmap (matplotlib.colors.ColorMap) - colormap for label_id

        parameters (dict) - {param name: param value} pairs

        fig (matplotlib.figures.Figure) - histogram figure

    """

    def __init__(self, values,
                 classify_on=None,
                 num_labels=3,
                 log=True,
                 cmap=None):
        """
        Instantiate classifier mapping <n> clusters to <num_labels>.

        Args:

            values (np.ndarray[float]) - basis for clustering

            classify_on (str or list) - attribute(s) on which to cluster

            num_labels (int) - number of class labels

            log (bool) - if True, cluster log-transformed values

            cmap (matplotlib.colors.ColorMap) - colormap for cell labels

        """

        # set values, whether to log transform them, and number of clusters
        self._values = values
        self.log = log

        self.num_labels = num_labels

        # set colormap
        self.set_cmap(cmap=cmap)

        # store parameters
        if type(classify_on) == str:
            classify_on = [classify_on]
        self.classify_on = classify_on
        self.parameters = dict(num_labels=num_labels,
                               log=log,
                               classify_on=classify_on)
        self.fig = None

    def __call__(self, data):
        """ Return labels for measurement <data>. """
        x =  data[self.classify_on].values
        if self.log:
            x = np.log(x)
        return self.classifier(x)

    @classmethod
    def from_measurements(cls, data, classify_on, **kwargs):
        """
        Fit classifier to data.

        Args:

            data (pd.DataFrame) - measurement data

            classify_on (str or list) - attribute(s) on which to cluster

            kwargs: keyword arguments for classifier

        Returns:

            classifier (Classifier derivative)

        """
        return cls(data[classify_on].values, classify_on, **kwargs)

    @classmethod
    def from_grouped_measurements(cls,
                            data,
                            classify_on,
                            groupby=None,
                            **kwargs):
        """
        Fit classifier to data grouped by a specified feature.

        Args:

            data (pd.DataFrame) - measurement data

            groupby (str) - attribute used to group measurement data

            classify_on (str or list) - attribute(s) on which to cluster

            kwargs: keyword arguments for classifier

        Returns:

            classifier (Classifier derivative)

        """

        if groupby is None:
            groupby = ('disc_genotype', 'disc_id', 'layer', 'im_label')
        values = df.groupby(by=groupby)[classify_on].mean().values
        return cls(values, classify_on, **kwargs)

    def save(self, dirpath, image=True):
        """
        Save classifier to specified path.

        Args:

            dirpath (str) - directory in which classifier is to be saved

            image (bool) - if True, save labeled histogram image

        """

        # create directory for classifier
        path = join(dirpath, 'classifier')
        if not exists(path):
            mkdir(path)

        # save values
        np.save(join(path, 'values.npy'), self._values)

        # save parameters
        io = IO()
        io.write_json(join(path, 'parameters.json'), self.parameters)

        # save image
        if image:

            # visualize classification
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

            classifier (Classifier derivative)

        """
        io = IO()
        values = io.read_npy(join(path, 'values.npy'))
        parameters = io.read_json(join(path, 'parameters.json'))
        return cls(values, **parameters)

    def show(self):
        """ Visualize classification. """
        pass

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

    def build_colormap(self, cmap, vmin=-1):
        """
        Build normalized colormap for class labels.

        Args:

            cmap (matplotlib.colormap)

            vmin (float) - lower bound for colorscale

        Returns:

            colormap (func) - function mapping class labels to colors

        """
        norm = Normalize(vmin=vmin, vmax=self.num_labels-1)
        return lambda x: cmap(norm(x))

    def build_classifier(self):
        """
        Build function that returns the most probable label for each of a series of values.
        """

        # build classifier that maps model components to labels.
        component_to_label = np.vectorize(self.component_to_label.get)

        def classifier(values):
            """ Returns <label> for <values>.  """
            return component_to_label(self.predict(values))

        return classifier
