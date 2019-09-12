import numpy as np
from sklearn.cluster import KMeans

from .classifiers import Classifier


class KMeansClassifier(Classifier):
    """
    K-means classifier.

    Attributes:

        groups (dict) - {cluster_id: label_id} pairs for merging clusters

        component_to_label (vectorized func) - maps cluster_id to label_id

        km (sklearn.cluster.KMeans) - kmeans object

        classifier (vectorized func) - maps value to label_id

        labels (np.ndarray[int]) - predicted labels

    Inherited attributes:

        values (array like) - basis for clustering

        attribute (str or list) - attribute(s) on which to cluster

        log (bool) - indicates whether clustering performed on log values

        cmap (matplotlib.colors.ColorMap) - colormap for label_id

        parameters (dict) - {param name: param value} pairs

        fig (matplotlib.figures.Figure) - histogram figure

    """

    def __init__(self, values,
                 num_components=3,
                 groups=None,
                 log=True,
                 **kwargs):
        """
        Instantiate k-means classifier.

        Args:

            values (array like) - basis for clustering

            num_components (int) - number of clusters

            groups (dict) - {cluster_id: label_id} pairs for merging clusters

            log (bool) - indicates whether clustering performed on log values

            kwargs: keyword arguments for Classifier parent class

        """

        # set groups and number of clusters
        if groups is None:
            groups = {k: k for k in range(num_components)}
        else:
            groups = {int(k): v for k, v in groups.items()}
        num_labels = len(groups)

        # instantiate classifier
        super().__init__(values, num_labels=num_labels, log=log, **kwargs)
        self.num_components = num_components
        self.component_to_label = np.vectorize(groups.get)
        self.groups = groups

        # build classifiers
        self.model = self.fit(self.values, self.num_components)
        self.classifier = self._build_value_to_groups_classifier()

        # assign group labels
        self.labels = self.classifier(self.values.reshape(-1, 1))

        # store parameters
        self.parameters.update(dict(groups=self.groups))

    @property
    def means(self):
        """ Mean of each cluster. """
        return self.model.cluster_centers_.ravel()

    def predict(self, values):
        """ Predict which component each of <values> belongs to. """
        return self.model.predict(values)

    @staticmethod
    def fit(values, n):
        """ Fit n clusters to x """
        return KMeans(n).fit(values.reshape(-1, 1))

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
        value_to_cluster = self._build_value_to_cluster_classifier(self.model)
        classifier = lambda x: self.component_to_label(value_to_cluster(x))
        return classifier
