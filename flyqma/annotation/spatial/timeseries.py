from math import floor
import numpy as np
import scipy.stats as st
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import warnings


def savgol(x, window_size=100, polyorder=1):
    """
    Perform Savitzky-Golay filtration of 1-D array.

    Args:

        x (np.ndarray) - ordered samples

        window_size (int) - filter size

        polyorder (int) - polynomial order

    Returns:

        trend (np.ndarray) - smoothed values

    """

    # window size must be odd
    if window_size % 2 == 0:
        window_size += 1
    window_size = int(window_size)

    with warnings.catch_warnings():

        # filter multidimensional indexing warning (fixed in scipy v1.2.0)
        warnings.filterwarnings('ignore', category=FutureWarning)

        # filter scipy gelsd warning
        warnings.filterwarnings('ignore', message='^internal gelsd')

        # apply savgol filter
        trend = savgol_filter(x, window_size, polyorder=polyorder)

    return trend


def get_running_mean(x, window_size=100):
    """
    Returns running mean for a 1D vector. This is the fastest implementation, but is limited to one-dimensional arrays and doesn't permit interval specification.

    Args:

        x (np.ndarray) - ordered samples, length N

        window_size (int) - size of window, W

    Returns:

        means (np.ndarray) - moving average of x

    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def get_rolling_window(x, window_size=100, resolution=1):
    """
    Return array slices within a rolling window.

    Args:

        x (np.ndarray) - ordered samples, length N

        window_size (int) - size of window, W

        resolution (int) - sampling interval

    Returns:

        windows (np.ndarray) - sampled values, N/resolution x W

    """

    if resolution is None:
        resolution = window_size

    shape = x.shape[:-1] + (floor((x.shape[-1] - window_size + 1)/resolution), window_size)
    strides = x.strides[:-1] + (x.strides[-1]*resolution,) + (x.strides[-1],)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def get_rolling_mean(x, **kw):
    """
    Compute rolling mean. This implementation permits flexible sampling intervals and multi-dimensional time series, but is slower than get_running_mean for 1D time series.

    Args:

        x (np.ndarray) - ordered samples, length N

        kw: arguments for window specification

    Returns:

        means (np.ndarray) - moving average of x, N/resolution x 1

    """
    return get_rolling_window(x, **kw).mean(axis=-1)


def get_binned_mean(x, window_size=100):
    """
    Returns mean values within non-overlapping sequential windows.

    Args:

        x (np.ndarray) - ordered samples, length N

        window_size (int) - size of window, W

    Returns:

        means (np.ndarray) - bin means, N/W x 1

    """
    return get_rolling_mean(x, window_size=window_size, resolution=window_size)


def apply_custom_roller(func, x, **kwargs):
    """
    Apply function to rolling window.

    Args:

        func (function) - function applied to each window, returns 1 x N_out

        x (np.ndarray) - ordered samples, length N

        kwargs: keyword arguments for window specification

    Returns:

        fx (np.ndarray) - function output for each window, N/resolution x N_out

    """
    windows = get_rolling_window(x, **kwargs)
    return np.apply_along_axis(func, axis=-1, arr=windows)


def subsample(x, frac=1):
    """
    Subsample array with replacement.

    Args:

        x (np.ndarray) - ordered samples, length N

        frac (float) - sample size (fraction of array)

    Returns:

        sample (np.ndarray) - subsampled values

    """
    return x[np.random.randint(0, len(x), size=floor(frac*len(x)))]


def bootstrap(x, func=np.mean, confidence=95, N=1000):
    """
    Returns point estimate obtained by parametric bootstrap.

    Args:

        x (np.ndarray) - ordered samples, length N

        func (function) - function applied to each bootstrap sample

        confidence (float) - confidence interval, between 0 and 100

        N (int) - number of bootstrap samples

    Returns:

        interval (np.ndarray) - confidence interval bounds

    """
    point_estimates = [func(subsample(x)) for _ in range(N)]
    q = (((100-confidence)/2), (100+confidence)/2)
    return np.percentile(point_estimates, q=q)


def get_rolling_mean_interval(x,
                              window_size=100,
                              resolution=1,
                              confidence=95,
                              nbootstraps=1000):
    """
    Evaluate confidence interval for moving average of ordered values.

    Args:

        x (np.ndarray) - ordered samples, length N

        window_size (int) - size of window, W

        resolution (int) - sampling interval

        confidence (float) - confidence interval, between 0 and 100

        nbootstraps (int) - number of bootstrap samples

    Returns:

        interval (np.ndarray) - confidence interval bounds, N/resolution x 2

    """

    # define bootstrapping operation
    f = lambda s: bootstrap(s, np.mean, confidence=confidence, N=nbootstraps)

    # apply to moving windows
    kw = dict(window_size=window_size, resolution=resolution)
    interval = apply_custom_roller(f, x, **kw)

    return interval


def get_rolling_gaussian(x, window_size=100, resolution=10):
    """
    Returns gaussian fit within sliding window.

    Args:

        x (np.ndarray) - ordered samples

        window_size (int) - size of window

        resolution (int) - sampling interval

    Returns:

        model (scipy.stats.norm)

    """

    # assemble sliding windows
    kw = dict(window_size=window_size, resolution=resolution)
    windows = get_rolling_window(x, **kw)

    # fit gaussians
    loc = windows.mean(axis=-1)
    scale = np.sqrt(windows.var(axis=-1))
    model = st.norm(loc=loc, scale=scale)

    return model


def detrend_signal(x, window_size=99, order=1):
    """
    Detrend and scale fluctuations using first-order univariate spline.

    Args:

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

    # evaluate trend
    trend = savgol(x, window_size=window_size, polyorder=order)

    # evaluate residuals (catch warnings for invalid values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        residuals = x - trend

    return residuals, trend


def smooth(x, window):
    """ Returns smoothed moving average of <x> within <window>. """
    kw = dict(window_size=window, resolution=int(window/5))
    return savgol(get_rolling_mean(x, **kw), window)


def plot_mean(ax, x, y,
              label=None,
              ma_type='sliding',
              window_size=100,
              resolution=1,
              line_color='k',
              line_width=1,
              line_alpha=1,
              linestyle=None,
              markersize=2,
              smooth=False,
              **kw):
    """
    Plot moving average.

    Args:

        x, y (array like) - timeseries data

        ax (matplotlib.axes.AxesSubplot) - axis which to which line is added

        label (str) - data label

        ma_type (str) - type of average used, either sliding, binned, or savgol

        window_size (int) - size of window

        resolution (int) - sampling interval

        line_color, line_width, line_alpha, linestyle - formatting parameters

        smooth (bool) - if True, apply secondary savgol filter

    Returns:

        line (matplotlib.lines.Line2D)

    """

    # get moving average (skip first point to avoid outliers)
    if ma_type == 'savgol':
        x_av = x[1:]
        y_av = savgol(y, window_size=window_size, polyorder=1)[1:]
    else:
        if ma_type == 'binned':
            resolution = window_size
        x_av = get_rolling_mean(x, window_size=window_size, resolution=resolution)
        y_av = get_rolling_mean(y, window_size=window_size, resolution=resolution)

    # get line and dashstyles
    if linestyle == None:
        linestyle = 'solid'
    dashstyles = {'solid': (None, None), 'dashed': (2.0, 2.0)}
    dashstyle = dashstyles[linestyle]

    # apply secondary smoothing (for visualization)
    if smooth:
        sw = int(window_size/3)
        if sw > 1:
            y_av = savgol(y_av, window_size=sw, polyorder=1)

    # plot line
    line = ax.plot(x_av, y_av,
                   linestyle=linestyle, dashes=dashstyle,
                   lw=line_width, color=line_color, alpha=line_alpha,
                   label=label, markersize=markersize)

    return line


def plot_mean_interval(ax, x, y,
                       ma_type='sliding',
                       window_size=100,
                       resolution=10,
                       nbootstraps=1000,
                       confidence=95,
                       color='grey',
                       alpha=0.25,
                       error_bars=False,
                       lw=0.):
    """
    Adds confidence interval for line average (sliding window or binned) to existing axes.

    Args:

        x, y (array like) - data

        ax (axes) - axis which to which line is added

        ma_type (str) - type of average used, either 'sliding' or 'binned'

        window_size (int) - size of sliding window or bin (num of cells)

        interval_resolution (int) - sampling resolution for confidence interval

        nbootstraps (int) - number of bootstraps

        confidence (float) - confidence interval, between 0 and 100

        color, alpha - formatting parameters

    """

    if ma_type == 'binned':
        resolution = window_size
    x_av = get_rolling_mean(x, window_size=window_size, resolution=resolution)
    y_av = get_rolling_mean(y, window_size=window_size, resolution=resolution)
    interval = get_rolling_mean_interval(y, window_size=window_size, resolution=resolution, nbootstraps=nbootstraps, confidence=confidence)
    y_lower, y_upper = interval.T
    _ = ax.fill_between(x_av, y_lower, y_upper,
                        color=color, alpha=alpha, lw=lw)

    if error_bars == True:
        ax.errorbar(x_av, y_av, yerr=[y_av-y_lower, y_upper-y_av], fmt='-o', color=color)

