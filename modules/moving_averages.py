__author__ = 'Sebi'

from math import floor
import numpy as np
import scipy.stats as st
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from functools import reduce
import array


def get_rolling_window(x, window_size, resolution=1):
    """
    Returns array slices for rolling window.

    Args:
    x (array) - array for which slices are desired, length N
    window_size (int) - size of rolling window, W
    resolution (int) - sampling interval

    Returns:
    windows (array) - sampled values from rolling windows, N/resolution x W
    """
    shape = x.shape[:-1] + (floor((x.shape[-1] - window_size + 1)/resolution), window_size)
    strides = x.strides[:-1] + (x.strides[-1]*resolution,) + (x.strides[-1],)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def savgol_smoothing(x, window_size=100, polyorder=1):
    if window_size % 2 == 0:
        window_size += 1
    window_size = int(window_size)
    trend = savgol_filter(x, window_length=window_size, polyorder=polyorder)
    return trend


def get_rolling_mean(x, **windows):
    """
    Compute rolling mean. This implementation permits variable sampling intervals and multi-dimensional time series, but is
    marginally slower than get_running_mean for 1D time series.

    Args:
    x (array) - array for which slices are desired, length N
    windows: arguments for window specification

    Returns:
    mean (array) - moving average of x, N/resolution x 1
    """
    return get_rolling_window(x, **windows).mean(axis=-1)


def get_binned_mean(x, window_size=100):
    """
    Returns mean value for sequential windows defined by number of cells.

    Parameters:
        x (array like) - time-ordered values
        window_size (int) - size of window
    """
    return get_rolling_mean(x, window_size=window_size, resolution=window_size)


def apply_custom_roller(func, x, **windows):
    """
    Applies custom function to rolling window.

    Args:
    func (function) - function applied to values within each window, returns 1 x (output_dim)
    x (array) - array for which slices are desired, length N
    windows: arguments for window specification

    Returns:
    func_eval (array) - function evaluation upon each window of x, N/resolution x (output_dim)
    """
    return np.apply_along_axis(func, axis=-1, arr=get_rolling_window(x, **windows))


def parametric_bootstrap(sample, func=np.mean, confidence=95, nbootstraps=1000):
    """
    Returns point estimate obtained by parametric bootstrap.

    Args:
    sample (array like) - sample values
    func (function) - function applied to each bootstrap sample
    confidence (float) - confidence interval desired, between 0 and 100
    nbootstraps (int) - number of bootstrap samples used

    Returns:
    interval (array) - lower and upper confidence interval bounds for point estimate
    """
    point_estimates = [func(subsample(sample)) for _ in range(nbootstraps)]
    interval = np.percentile(point_estimates, q=(((100-confidence)/2), (100+confidence)/2))
    return interval


def subsample(x, frac=1):
    """
    Subsamples array with replacement.

    Args:
    x (array) - array for which slices are desired, length N
    frac (float) - fraction of array size to be returned

    Returns:
    sample (array) - subsampled values
    """
    return x[np.random.randint(0, len(x), size=floor(frac*len(x)))]


def get_rolling_mean_interval(x, window_size=100, resolution=1, confidence=95, nbootstraps=1000):
    """
    Computes confidence interval for rolling average of a time series.

    Args:
    x (array) - array for which slices are desired, length N
    window_size (int) - size of rolling window, W
    resolution (int) - sampling interval
    confidence (float) - confidence interval desired, between 0 and 100
    nbootstraps (int) - number of bootstrap samples used

    Returns:
    interval (array) - lower and upper bounds on confidence interval for bootstrapped point estimate, N/resolution x 2
    """
    func = lambda sample: parametric_bootstrap(sample, func=np.mean, confidence=confidence, nbootstraps=nbootstraps)
    interval = apply_custom_roller(func, x, window_size=window_size, resolution=resolution)
    return interval


def get_rolling_gaussian(x, window_size=100, resolution=10):
    """
    Returns gaussian fit for rolling window.

    Args:
    x (array) - array for which slices are desired, length N
    window_size (int) - size of sliding window
    resolution (int) - sampling resolution, e.g. 2 constructs a window starting from every other cell

    Returns:
    model (scipy.stats.norm instance)
    """
    windows = get_rolling_window(x, window_size=window_size, resolution=resolution)
    model = st.norm(loc=windows.mean(axis=-1), scale=np.sqrt(windows.var(axis=-1)))
    return model


def get_running_mean(x, window_size=100):
    """
    Returns running mean for a 1D vector. This is currently the fastest implementation, but is currently limited to
    one-dimensional arrays and doesn't permit interval specification.

    Args:
    x (array) - array for which slices are desired
    window_size (int) - size of rolling window

    Returns:
    mu_x (array) - moving average of x
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def plot_mean(x, y, ax, label=None, ma_type='sliding', window_size=100, resolution=1, line_color='black', line_width=1, line_alpha=1, linestyle=None, markersize=2, smooth=False):
    """
    Adds line average (sliding window or binned) to existing axes.

    Parameters:
        x, y (array like) - data
        ax (axes) - axis which to which line is added
        label (str) - data label for legend
        ma_type (str) - type of average used, either sliding, binned, or savgol
        window_size (int) - size of sliding window or bin (num of cells)
        line_color, line_width, line_alpha, linestyle - formatting parameters
        extra_smoothing (bool) - if True, apply secondary savgol filter

    Returns:
        line (Line2D object)
    """

    dashd = {'solid': (None, None), 'dashed': (2.0, 2.0)}

    if ma_type == 'savgol':
        x_av = x
        y_av = savgol_smoothing(y, window_size=window_size, polyorder=1)

    else:
        if ma_type == 'binned':
            resolution = window_size
        x_av = get_rolling_mean(x, window_size=window_size, resolution=resolution)
        y_av = get_rolling_mean(y, window_size=window_size, resolution=resolution)

    if linestyle == None:
        linestyle = 'solid'

    if smooth:
        smoothing_window = int(window_size/5)
        if smoothing_window > 1:
            y_av = savgol_smoothing(y_av, window_size=smoothing_window, polyorder=1)

    line = ax.plot(x_av, y_av, linestyle=linestyle, dashes=dashd[linestyle], linewidth=line_width, color=line_color, alpha=line_alpha, label=label, markersize=markersize)
    return line


def plot_mean_interval(x, y, ax, ma_type='sliding', window_size=100, resolution=1, nbootstraps=1000, confidence=95, color='grey', alpha=0.25, error_bars=False, lw=0.):
    """
    Adds confidence interval for line average (sliding window or binned) to existing axes.

    Parameters:
        x, y (array like) - data
        ax (axes) - axis which to which line is added
        ma_type (str) - type of average used, either 'sliding' or 'binned'
        window_size (int) - size of sliding window or bin (num of cells)
        interval_resolution (int) - sampling resolution for confidence interval
        nbootstraps (int) - number of bootstraps
        confidence (float) - confidence interval, between 0 and 100
        color, alpha - formatting parameters
    """
    #nbootstraps = 250
    #resolution=1

    if ma_type == 'binned':
        resolution = window_size
    x_av = get_rolling_mean(x, window_size=window_size, resolution=resolution)
    y_av = get_rolling_mean(y, window_size=window_size, resolution=resolution)
    interval = get_rolling_mean_interval(y, window_size=window_size, resolution=resolution, nbootstraps=nbootstraps, confidence=confidence)
    y_lower, y_upper = interval.T
    _ = ax.fill_between(x_av, y_lower, y_upper, color=color, alpha=alpha, lw=lw)

    if error_bars == True:
        ax.errorbar(x_av, y_av, yerr=[y_av-y_lower, y_upper-y_av], fmt='-o', color=color)


class Roller:
    def __init__(self, cells, window_size=100, resolution=10):
        self.df = cells.df
        self.add_disc_ids()
        self.window_size = window_size
        self.resolution = resolution

    def add_disc_ids(self):
        """ Preallocated disc_id boolean arrays. """
        unique_ids = self.df.disc_id.unique()
        for _id in unique_ids:
            self.df['d_'+str(_id)] = self.df.disc_id == _id

    def apply_custom_roller(self, f, x):
        """ Applies custom function to rolling window."""
        return np.apply_along_axis(f, axis=-1, arr=get_rolling_window(x, self.window_size, self.resolution))

    def parametric_bootstrap(self, idx, variable='green', nbootstraps=1000):
        """ Returns point estimate obtained by parametric bootstrap. """

        # create dictionary of values for each disc
        cells = self.df.iloc[idx]
        ids = cells.disc_id.unique()
        #discs = {i: cells[cells.disc_id==_id][variable].values for i, _id in enumerate(unique_ids)}

        #discs = [cells[cells.disc_id==_id][variable].values for _id in ids]
        discs = [cells[cells['d_'+str(_id)]][variable].values for _id in ids]
        p = [len(disc)/len(cells) for disc in discs]

        # compute point estimates for each subsample
        point_estimates = [(self.subsample_discs(discs, p, ids.size)).mean() for _ in range(nbootstraps)]

        return point_estimates

    def get_rolling_mean(self, variable='t', nbootstraps=1000):
        """ Computes confidence interval for rolling average of a time series."""
        strap = lambda idx: self.parametric_bootstrap(idx, variable, nbootstraps)
        indices = np.arange(self.df.shape[0])
        point_estimates = self.apply_custom_roller(strap, indices)
        return np.percentile(point_estimates, q=50, axis=1)

    def get_rolling_interval(self, variable='green', confidence=95, nbootstraps=1000):
        """ Computes confidence interval for rolling average of a time series."""

        # get point estimates
        strap = lambda idx: self.parametric_bootstrap(idx, variable, nbootstraps)
        indices = np.arange(self.df.shape[0])
        point_estimates = self.apply_custom_roller(strap, indices)

        # use point estimates to construct confidence interval
        interval = np.percentile(point_estimates, q=(((100-confidence)/2), (100+confidence)/2), axis=1)

        return interval

    def plot_interval(self, variable='green', ax=None, nbootstraps=1000, confidence=95, color='k', alpha=0.5):

        #t = self.get_rolling_mean('t', nbootstraps)
        t = get_rolling_mean(self.df.t, window_size=self.window_size, resolution=self.resolution)

        low, high = self.get_rolling_interval(variable, confidence, nbootstraps)

        if ax is None:
            fig, ax = plt.subplots()

        ax.fill_between(t, low, high, color=color, alpha=alpha)


def detrend_signal(x, window_size=99, order=1):
    """
    Detrend and scale fluctuations using first-order univariate spline.

    Parameters:
        x (np array) -ordered samples
        window_size (int) - size of interpolation window for lowpass filter
        order (int) - spline order

    Returns:
        residuals (np array) - detrended residuals
        trend (np array) - spline fit to signal
    """

    # use odd window size
    if window_size % 2 == 0:
        window_size += 1
    window_size = int(window_size)

    trend = savgol_smoothing(x, window_size=window_size, polyorder=order)
    residuals = x - trend
    return residuals, trend
