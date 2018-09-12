import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.cluster import k_means


class KDE:

    def __init__(self, df, bandwidth=100, n=2):
        self.xy = df[['centroid_x', 'centroid_y']].values
        self.bandwidth = bandwidth
        self.kde = self.fit_kde(self.xy, bandwidth)
        self.density = self.evaluate_density()
        self.cluster_density(n)
        self.evaluate_cluster_threshold()

    def __call__(self, x):
        return np.exp(self.kde.score_samples(x))

    @staticmethod
    def fit_kde(xy, bandwidth=100):
        """ Fit gaussian kernel to xy coordinates spatial density."""
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(xy)
        return kde

    def evaluate_density(self):
        return self.__call__(self.xy)

    def cluster_density(self, n=2):
        values = np.log10(self.density.reshape(-1, 1))
        centroid, label, inertia = k_means(values, n)
        self.centroid = centroid
        self.label = label

    def evaluate_cluster_threshold(self, cluster_id=0):
        order = np.argsort(self.centroid.flatten())
        threshold = self.density[self.label==order[0]].max()
        self.threshold = threshold
        self.mask = self.density > threshold
