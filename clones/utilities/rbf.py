from sklearn.cluster import k_means
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt


class RBF:
    """
    Radial basis function fit to scalar field.
    """

    def __init__(self, xy, z, smooth=1, function='multiquadric'):
        """
        Instantiate RBF object.self

        Args:

            xy (2D np.ndarray[float]) - node coordinates (2xN)

            z (1D np.ndarray[float]) - scalar values

            smooth (float) - RBF smoothing parameter

            function (str) - RBF kernel

        """
        self.xy = xy
        self.z = z
        self.rbf = self.fit(smooth, function)
        self.evaluate()
        self.cluster()

    def __call__(self, *args):
        return self.rbf(*args)

    def fit(self,
            smooth=1,
            function='multiquadric'):
        """ Fit RBF """
        rbf = interpolate.rbf.Rbf(*self.xy, self.z, smooth=smooth, function=function)
        return rbf

    def evaluate(self):
        self.zi = self.__call__(*self.xy)

    def cluster(self, n=3):
        centroids, labels, inertia = k_means(self.zi.reshape(-1, 1), n)
        self.labels = np.empty_like(labels, dtype=int)
        self.thresholds = np.empty_like(centroids.flatten())
        indices = np.argsort(centroids.flatten())
        for label, ind in enumerate(indices):
            self.labels[labels==ind] = label
            self.thresholds[label] = self.zi[labels==ind].max()
        self.centroids = centroids[indices]

    def plot_points(self,
                    ax,
                    s=10,
                    lw=0,
                    cmap=None):
        if cmap is None:
            cmap = plt.cm.Reds
        y, x = self.xy
        vmin, vmax = self.z.min(), self.z.max()
        ax.scatter(x, y, c=self.zi, vmin=vmin, vmax=vmax, cmap=cmap, s=s, linewidths=lw)
        ax.axis('off')

    def show(self):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot(self.z, self.zi, '.k')
        ax.set_xlabel('Data')
        ax.set_ylabel('Interpolated')
