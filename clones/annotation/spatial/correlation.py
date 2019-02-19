__author__ = 'Sebastian Bernasek'

import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from ...visualization.settings import *
from .timeseries import plot_mean, plot_mean_interval
from .timeseries import smoothed_moving_average


class CorrelationProperties:

    # fraction of cells used to determine window size
    window_frac = 0.01

    @property
    def num_measurements(self):
        """ Estimated number of measurements. """
        return np.sqrt(self.d_ij.size)

    @property
    def window(self):
        """ Window size. """
        return int(np.round((self.num_measurements * self.window_frac) ** 2))

    @property
    def d_av(self):
        """ Smoothed moving average distances. """
        return smoothed_moving_average(self.d_ij, self.window)

    @property
    def C_av(self):
        """ Smoothed moving average correlation. """
        return smoothed_moving_average(self.C_ij, self.window)

    @property
    def characteristic_length(self):
        """ Characteristic length over which correlations decay. """
        return CharacteristicLength(self).characteristic_length


class CorrelationData(CorrelationProperties):
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

    @property
    def num_fluctuations(self):
        """ Number of pairwise fluctuations. """
        return len(self.d_ij)

    @property
    def ubound(self):
        """ Upper bound on pairwise distances. """
        return np.percentile(self.d_ij, 99)

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

    @default_figure
    def histogram_distances(self, ax=None, **kwargs):
        """ Plot histogram of pairwise distances between measurements. """
        _ = ax.hist(self.d_ij, **kwargs)
        ax.set_ylabel('Num. pairs')
        ax.set_xlabel('Separation distance')

    @default_figure
    def histogram_fluctuations(self, ax=None, **kwargs):
        """ Plot histogram of pairwise fluctuations between measurements. """
        _ = ax.hist(self.C_ij, **kwargs)
        ax.set_ylabel('Num. pairs')
        ax.set_xlabel('Fluctuation')

    @default_figure
    def visualize(self,
                  null_model=False,
                  scatter=False,
                  confidence=False,
                  zero=True,
                  smooth=True,
                  ma_type='sliding',
                  window_size=500,
                  resolution=100,
                  nbootstraps=50,
                  color='k',
                  max_distance=None,
                  ax=None):
        """
        Plot pairwise normalized fluctuations versus pairwise distances.

        Args:

            null_model (bool) - if True, shuffle d_ij vector

            scatter (bool) - if True, show individual markers

            confidence (bool) - if True, include confidence interval

            zero (bool) - if True, include zero correlation line for reference

            smooth (bool) - if True, apply smoothing to moving average

            ma_type (str) - type of average used, either sliding, binned, or savgol

            window_size (int) - size of window

            resolution (int) - sampling interval

            nbootstraps (int) - number of bootstrap samples for confidence interval

            color (str) - color used for confidence interval

            max_distance (float) - largest pairwise distance included

            ax (mpl.axes.AxesSubplot) - if None, create figure

        Returns:

            ax (mpl.axes.AxesSubplot)

        """

        d = self.d_ij
        C = self.C_ij

        # if null_model is True, randomly shuffle d_ij vector
        if null_model:
            d = np.random.choice(d, self.num_fluctuations, replace=False)

            # sort by distance
            ind = np.argsort(d)
            d = d[ind]
            C = C[ind]

        # filter by max_distance
        if max_distance is None:
            max_distance = self.ubound
        xmax = (max_distance//100)*100
        ind = (d<=max_distance)
        C = C[ind]
        d = d[ind]

        # get smoothing arguments
        ma_kw=dict(ma_type=ma_type, window_size=window_size, resolution=resolution)

        # plot moving average
        plot_mean(ax=ax, x=d, y=C,
                  line_color=color,
                  line_width=1,
                  line_alpha=1,
                  markersize=2,
                  smooth=smooth,
                  **ma_kw)

        # plot confidence interval for moving average
        if confidence:
            plot_mean_interval(ax=ax, x=d, y=C,
                               confidence=95,
                               color=color,
                               alpha=0.35,
                               nbootstraps=nbootstraps,
                               **ma_kw)

        # plot markers
        if scatter:
            ax.scatter(d, C, alpha=0.2, color='grey', linewidth=0)

        # plot zero reference line
        if zero:
            ax.plot([0, xmax], np.zeros(2), '-r', linewidth=1, alpha=0.25)

        # format
        ymin, ymax = -0.15, 0.7
        ax.set_ylim(ymin, ymax), ax.set_yticks([-0., 0.2, 0.4, .6])
        ax.set_xlim(0, xmax)
        ax.set_xlabel('Pairwise distance')
        ax.set_ylabel('Correlation')


class SpatialCorrelation(CorrelationData):
    """
    Object for evaluating spatial correlation of expression between measurements.

    Attributes:

        attribute (str) - name of correlated attribute

        log (bool) - if True, log-transform attribute values

    Inherited attributes:

        d_ij (np array) - pairwise separation distances between measurements

        C_ij (np array) - normalized pairwise fluctuations between measurements

    """

    def __init__(self, graph, attribute, log=False):
        """
        Instantiate object for evaluating spatial correlation of expression between cells.

        Args:

            graph (spatial.Graph derivative) - graph object

            attribute (str) - name of correlated attribute

            log (bool) - if True, log-transform attribute values

        """

        # store parameters
        self.attribute = attribute
        self.log = log

        # get distances vector
        d_ij = self.get_distances_vector(graph)

        # get covariance vector
        attribute_values = graph.df[attribute].values.reshape(-1, 1)
        if log:
            attribute_values = np.log(attribute_values)
        C_ij = self.get_covariance_vector(attribute_values)

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
    def get_distances_vector(cls, graph):
        """
        Get upper triangular portion of pairwise distance matrix.

        Args:

            graph (spatial.Graph derivative) - graph object

        Returns:

            distances (1D np.ndarray) - pairwise distances, ordered row then column

        """

        # compute pairwise distances between cells
        x, y = graph.node_positions_arr.T
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        x_component = np.repeat(x**2, x.size, axis=1) + np.repeat(x.T**2, x.size, axis=0) - 2*np.dot(x, x.T)
        y_component = np.repeat(y**2, y.size, axis=1) + np.repeat(y.T**2, y.size, axis=0) - 2*np.dot(y, y.T)

        # get upper triangular portion (excludes self edges and duplicates)
        distances = cls.get_matrix_upper(np.sqrt(x_component + y_component))

        return distances

    @classmethod
    def get_covariance_vector(cls, vector):
        """
        Get upper triangular portion of pairwise expression covariance matrix.

        Args:

            vector (1D np.ndarray) - attribute values for each measurement

        Returns:

            covariance (1D np.ndarray) - pairwise fluctuations, ordered row then column

        """

        # compute correlation matrix and return upper triangular portion
        v = (vector - vector.mean())
        covariance_matrix = np.dot(v, v.T) / np.var(vector)
        covariance = cls.get_matrix_upper(covariance_matrix)
        return covariance


class CharacteristicLength:
    """
    Class for determining the characteristic length over which correlations decay.
    """

    def __init__(self, correlation, fraction_of_max=0.1):
        """
        Args:

            correlation (SpatialCorrelation)

            fraction_of_max (float) - fraction of peak correlation used to fit exponential decay

        """

        self.fraction_of_max = fraction_of_max

        # extract correlation decay dynamics
        xmax, x, y = self.extract_decay(correlation, fraction_of_max)
        self.xmax = xmax
        self.x = x
        self.y = y

        # fit exponential decay
        popt, perr = self.fit(self.x_normed, self.y, self.model)
        self.popt = popt
        self.perr = perr

    @property
    def x_normed(self):
        """ Distance vector normalized by maximum value. """
        return self.x / self.xmax

    @property
    def yp(self):
        """ Predicted correlation values. """
        return self.model(self.x_normed, *self.popt)

    @property
    def characteristic_length(self):
        """ Characteristic decay length. """
        return self.xmax / self.popt[1]

    @staticmethod
    def extract_decay(correlation, fraction_of_max):
        """ Extract decay. """

        x = correlation.d_av
        y = correlation.C_av
        ymin = fraction_of_max * y.max()
        max_idx = (y < ymin).nonzero()[0][0]

        return x[max_idx], x[0:max_idx], y[0:max_idx]

    @staticmethod
    def model(x, a, b):
        """ Exponential decay model. """
        return a * np.exp(-b*x)

    @staticmethod
    def fit(x, y, model):
        """ Fit exponential decay model to decay vectors. """
        popt, pcov = curve_fit(model, x, y, p0=(1., 1.))
        perr = np.sqrt(np.diag(pcov))
        return popt, perr

    def plot_measured(self, ax, **kwargs):
        """ Plot measured correlation decay. """
        ax.plot(self.x, self.y, label='Measured', **kwargs)

    def plot_fit(self, ax, **kwargs):
        """ Plot model fit. """
        ax.plot(self.x, self.yp, label='Fit', **kwargs)

    @default_figure
    def plot(self, ax=None, **kwargs):
        """ Plot measured correlation decay alongside model fit. """
        self.plot_measured(ax, color='k', **kwargs)
        self.plot_fit(ax, color='r', **kwargs)
        ax.set_ylabel('Correlation')
        ax.set_xlabel('Distance')
