import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib.colors import ListedColormap, Normalize


class KM:

    def __init__(self, values, n=3, groups=None, log=False, cmap=None):

        # set groups and number of clusters
        if groups is None:
            groups = {k: k for k in range(n)}
        else:
            assert n == len(groups), 'Wrong number of groups.'
        self.n = n
        self.groups = groups
        self.cluster_to_group = np.vectorize(groups.get)

        # set values
        self.log = log
        if self.log:
            values = np.log10(values)
        self.x = values

        # build classifiers
        self.km = self._kmeans(self.x, self.n)
        self.classifier = self._build_value_to_groups_classifier()

        # assign group labels
        self.labels = self.classifier(self.x.reshape(-1, 1))

        # set colormap
        self.set_cmap(cmap)

    def __call__(self, x):
        return self.classifier(x)

    @staticmethod
    def _kmeans(x, n):
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

    def set_cmap(self, cmap=None):
        if cmap is None:
            cmap = plt.cm.plasma
            norm = Normalize(vmin=0, vmax=self.n)
            colors = [cmap(norm(i)) for i in range(self.n)]
            self.cmap = ListedColormap(colors)
        else:
            self.cmap = cmap

    def show(self, **kw):
        ax = self._show(self.x, self.labels, self.cmap, **kw)

    @staticmethod
    def _show(x, labels, cmap, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        bins = np.linspace(x.min(), x.max(), 50)
        for label in set(labels):
            xi = x[(labels==label)]
            ax.hist(xi, bins=bins, facecolor=cmap(label))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Values', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        return ax


class CellClassifier(KM):

    def __init__(self, values, classify_on='r_normalized', **kw):
        self.classify_on = classify_on
        KM.__init__(self, values, **kw)

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

    @staticmethod
    def _show(self, x, labels, cmap, ax=None):
        ax = KM._show(x, labels, cmap, ax=ax)
        ax.set_xlabel('RFP level', fontsize=8)
        ax.set_ylabel('Number of nuclei', fontsize=8)
        return ax


class CloneClassifier:
    """ Classifier for assigning genotypes to clonal subpopulations within an image layer. """

    def __init__(self, df, cell_classifier):
        self.im_to_genotype = self.build_classifier(df, cell_classifier)
        self.genotypes = self.__call__(df.im_label.values)

    def __call__(self, im_labels):
        return self.im_to_genotype(im_labels)

    @staticmethod
    def from_layer(layer, cell_classifier):
        return CloneClassifier(layer.df, cell_classifier)

    @staticmethod
    def get_mode(x):
        mode, count = Counter(x).most_common(1)[0]
        return mode

    @classmethod
    def build_classifier(cls, cells, cell_classifier):
        """
        Build classifier assigning genotypes to subpopulation IDs.

        Args:
        cells (pd.DataFrame) - cell data including subpopulation ID labels
        cell_classifier (callable) - assigns genotype to individual cell record
        """
        majority_vote = lambda x: cls.get_mode(cell_classifier(x))
        im_to_genotype_dict = cells.groupby('im_label').apply(majority_vote).to_dict()
        im_to_genotype_dict[-1] = -1
        return np.vectorize(im_to_genotype_dict.get)
