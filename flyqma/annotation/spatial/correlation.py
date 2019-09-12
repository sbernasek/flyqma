import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from ...visualization import *

from .timeseries import plot_mean, plot_mean_interval
from .timeseries import smooth


class CorrelationProperties:

    # fraction of cells used to determine window size
    window_frac = 0.01

    @property
    def num_fluctuations(self):
        """ Number of pairwise fluctuations. """
        return len(self.d_ij)

    @property
    def ubound(self):
        """ Upper bound on pairwise distances. """
        return np.percentile(self.d_ij, 99)

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
        return smooth(self.d_ij, self.window)

    @property
    def C_av(self):
        """ Smoothed moving average correlation. """
        return smooth(self.C_ij, self.window)

    @property
    def characteristic_length(self):
        """ Characteristic length over which correlations decay. """
        try:
            length = CharacteristicLength(self).characteristic_length
        except:
            length = None
        return length


class CorrelationVisualization:
    """ Visualization methods for SpatialCorrelation. """

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
            ax.plot([0, max_distance], [0, 0], '-r', linewidth=1, alpha=0.25)

        # format
        ymin, ymax = -0.15, 0.7
        ax.set_ylim(ymin, ymax), ax.set_yticks([-0., 0.2, 0.4, .6])
        ax.set_xlim(0, max_distance)
        ax.set_xlabel('Pairwise distance')
        ax.set_ylabel('Correlation')


class SpatialCorrelation(CorrelationProperties, CorrelationVisualization):
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
        Concatenate current instance with a second SpatialCorrelation instance.

        Args:

            correlation (SpatialCorrelation)

        Returns:

            self (SpatialCorrelation) - updated correlations

        """

        # concatenate distanced and fluctuations
        d_ij = np.hstack((self.d_ij, correlation.d_ij))
        C_ij = np.hstack((self.C_ij, correlation.C_ij))

        # sort by distance
        ind = np.argsort(d_ij)
        self.d_ij = d[ind]
        self.C_ij = C_ij[ind]

        return self


class CharacteristicLength:
    """
    Class for determining the characteristic length over which correlations decay.
    """

    def __init__(self, correlation, fraction_of_max=0.01):
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
