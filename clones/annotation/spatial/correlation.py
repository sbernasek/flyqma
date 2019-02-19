__author__ = 'Sebastian Bernasek'

import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt


class CorrelationData:
    """
    Container for correlations between 1-D timeseries.

    Attributes:

        d_ij (np array) - pairwise separation distances between measurements

        C_ij (np array) - normalized pairwise fluctuations between measurements

    """

    def __init__(self, d_ij=None, C_ij=None):
        """
        Instantiate container for correlations between 1-D timeseries.

        Args:

            d_ij (np.ndarray) - pairwise separation distances

            C_ij (np.ndarray) - normalized pairwise fluctuations

        """

        if d_ij is None:
            d_ij = np.empty(0)
        if C_ij is None:
            C_ij = np.empty(0)

        ind = np.argsort(d_ij)
        self.d_ij = d_ij[ind]
        self.C_ij = C_ij[ind]

    def __add__(self, correlation):
        """
        Concatenate current instance with a second CorrelationData instance.

        Args:

            correlation (analysis.correlation.CorrelationData)

        Returns:

            self (analysis.correlation.CorrelationData) - updated correlations

        """

        # concatenate distanced and fluctuations
        d_ij = np.hstack((self.d_ij, correlation.d_ij))
        C_ij = np.hstack((self.C_ij, correlation.C_ij))

        # sort by distance
        ind = np.argsort(d_ij)
        self.d_ij = d[ind]
        self.C_ij = C_ij[ind]

        return self

    @staticmethod
    def get_binned_stats(x, y, bins, statistic='mean'):
        """
        Group samples into x-bins and evaluate aggregate statistic of y-values.

        Args:

            x (np array) - values upon which samples are grouped

            y (np array) - values upon which aggregate statistics are evaluated

            bins (np array) - bin edges

            statistic (str) - aggregation statistic applied to each bin

        Returns:

            centers (np array) - bin centers

            stats (np array) - aggregate statistic for each bin

        """

        # if None, break into 10 intervals
        if bins is None:
            bins = np.arange(x.min(), x.max(), x.max()/10)

        # evaluate statistic
        stats, _, _ = binned_statistic(x, y, statistic=statistic, bins=bins)

        # compute bin centers
        centers = [bins[i]+(bins[i+1]-bins[i])/2 for i in range(0,len(bins)-1)]

        return centers, stats

    @classmethod
    def bootstrap(cls, x, y,
                  confidence=95,
                  N=1000,
                  bins=None):
        """
        Evaluate confidence interval for aggregation statistic.

        Args:

            x (np array) - values upon which samples are grouped

            y (np array) - values upon which aggregate statistics are evaluated

            N (int) - number of repeated samples

            confidence (int) - confidence interval, between 0 and 100

            bins (np array) - bins within which the statistic is applied

        Returns:

            centers (np array) - centers of distance bins

            uppers, lowers (np array) - statistic confidence interval bounds

        """

        # get indices for bootstrap resampling
        indices = np.random.randint(0, len(x), size=(N, len(x)))

        # evaluate aggregation statistic
        stats = []
        for ind in indices:
            centers, stat = cls.get_binned_stats(x[ind], y[ind], bins=bins)
            stats.append(stat)
        stats = np.array(stats)

        # evaluate confidence interval
        uppers = np.nanpercentile(stats, q=(100+confidence)/2, axis=0)
        lowers = np.nanpercentile(stats, q=(100-confidence)/2, axis=0)

        return centers, uppers, lowers

    # def visualize(self,
    #               ax=None,
    #               null_model=False,
    #               scatter=True,
    #               confidence=True,
    #               zero=True,
    #               ma_kw=None,
    #               nbootstraps=100,
    #               color='k',
    #               max_distance=500):
    #     """
    #     Plot pairwise normalized fluctuations versus pairwise distances.

    #     Args:

    #         ax (mpl.axes.AxesSubplot) - if None, create figure

    #         null_model (bool) - if True, shuffle d_ij vector

    #         scatter (bool) - if True, show individual markers

    #         confidence (bool) - if True, include confidence interval

    #         zero (bool) - if True, include zero correlation line for reference

    #         interval_kw (dict) - keyword arguments for interval formatting

    #         ma_kw (dict) - keyword arguments for moving average smoothing

    #         nbootstraps (int) - number of bootstrap samples for confidence interval

    #         color (str) - color used for confidence interval

    #         max_distance (float) - largest pairwise distance included

    #     Returns:

    #         ax (mpl.axes.AxesSubplot)

    #     """

    #     # create figure/axis
    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=(9, 3))

    #     # get number of pairwise fluctuations
    #     number_of_pairs = len(self.d_ij)

    #     # if null_model is True, randomly shuffle d_ij vector
    #     if null_model:
    #         d = np.random.choice(self.d_ij, number_of_pairs, replace=False)
    #     else:
    #         d = self.d_ij

    #     # filter by max_distance
    #     xmax = (max_distance//100)*100
    #     ind = (d<=max_distance)
    #     C = self.C_ij[ind]
    #     d = d[ind]

    #     # sort by distance
    #     ind = np.argsort(d)
    #     d = d[ind]
    #     C = C[ind]

    #     # get smoothing arguments
    #     if ma_kw is None:
    #         ma_kw=dict(ma_type='savgol', window_size=100, resolution=50)

    #     # plot moving average
    #     plot_mean(d, C, ax,
    #               line_color=color,
    #               line_width=1,
    #               line_alpha=1,
    #               markersize=2,
    #               **ma_kw)

    #     # plot confidence interval for moving average
    #     if confidence:
    #         plot_mean_interval(d, C, ax,
    #                            confidence=95,
    #                            color=color,
    #                            alpha=0.35,
    #                            nbootstraps=nbootstraps,
    #                            **ma_kw)

    #     # plot markers
    #     if scatter:
    #         ax.scatter(d, C, alpha=0.2, color='grey', linewidth=0)

    #     # plot zero reference line
    #     if zero:
    #         ax.plot([0, xmax], np.zeros(2), '-r', linewidth=1, alpha=0.25)

    #     # format
    #     ymin, ymax = -0.6, 0.6
    #     ax.set_ylim(ymin, ymax), ax.set_yticks([-0.5, 0, 0.5])
    #     ax.set_ylabel('mean corr.', fontsize=10)
    #     ax.set_xlabel('y-distance between cells $i, j$ (px)', fontsize=12)
    #     ax.tick_params(labelsize=10)
    #     ax.set_xlim(0, xmax)

    #     return ax


class SpatialCorrelation(CorrelationData):
    """
    Object for evaluating spatial correlation of expression between cells.

    Attributes:

        channel (str) - expression channel for which correlations are desired

        y_only (bool) - if True, only use y-component of pairwise distances

        raw (bool) - if True, use raw fluorescence intensities

    Inherited attributes:

        d_ij (np array) - pairwise separation distances between measurements

        C_ij (np array) - normalized pairwise fluctuations between measurements

    """

    def __init__(self,
                 df=None,
                 channel='green',
                 y_only=True,
                 raw=False):
        """
        Instantiate object for evaluating spatial correlation of expression between cells.

        Args:

            df (pd.Dataframe) - cell measurement data

            channel (str) - expression channel for which correlations are desired

            y_only (bool) - if True, only use y-component of pairwise distances

            raw (bool) - if True, use raw fluorescence intensities

        """

        # store parameters
        self.channel = channel
        self.y_only = y_only
        self.raw = raw

        # get pairwise distances and fluctuations
        if df is None:
            d_ij, C_ij = np.ndarray([]), np.ndarray([])
        else:

            # get distances vector
            d_ij = self.get_distances_vector(df, y_only=y_only)

            # get covariance vector
            expression = df[channel].values
            if raw is True and channel != 'ratio':
                expression *= df[cells.normalization].values
            C_ij = self.get_covariance_vector(expression.reshape(-1, 1))

        # instantiate parent object
        CorrelationData.__init__(self, d_ij, C_ij)

    @staticmethod
    def get_matrix_upper(matrix):
        """
        Return upper triangular portion of a 2-D matrix.

        Parameters:

            matrix (2D np.ndarray)

        Returns:

            upper (1D np.ndarray) - upper triangle, ordered row then column

        """
        return matrix[np.triu_indices(len(matrix), k=1)]

    @classmethod
    def get_distances_vector(cls, df, y_only=False):
        """
        Get upper triangular portion of pairwise distance matrix.

        Args:

            df (pd.Dataframe) - cell measurements including position data

            y_only (bool) - if True, only use y-component of cell positions

        Returns:

            distances (1D np.ndarray) - pairwise distances, ordered row then column

        """

        # if no measurements are included, return None
        if len(df) == 0:
            return None

        # compute pairwise distances between cells
        x = df.centroid_x.values.reshape((len(df), 1))
        y = df.centroid_y.values.reshape((len(df), 1))
        x_component = np.repeat(x**2, x.size, axis=1) + np.repeat(x.T**2, x.size, axis=0) - 2*np.dot(x, x.T)
        if y_only is True:
            x_component *= 0
        y_component = np.repeat(y**2, y.size, axis=1) + np.repeat(y.T**2, y.size, axis=0) - 2*np.dot(y, y.T)

        # get upper triangular portion (excludes self edges and duplicates)
        distances = cls.get_matrix_upper(np.sqrt(x_component + y_component))

        return distances

    @classmethod
    def get_covariance_vector(cls, vector):
        """
        Get upper triangular portion of pairwise expression covariance matrix.

        Args:

            vector (1D np.ndarray) - expression levels for each cell

        Returns:

            covariance (1D np.ndarray) - pairwise fluctuations, ordered row then column

        """

        # if vector is of length zero, return None
        if len(vector) == 0:
            return None

        # compute correlation matrix and return upper triangular portion
        v = (vector - vector.mean())
        covariance_matrix = np.dot(v, v.T) / np.var(vector)
        covariance = cls.get_matrix_upper(covariance_matrix)
        return covariance

