import numpy as np
from collections import Counter


def resample_uniformly(x, y, size=None, cutoff=None):
    """
    Resample X and Y uniformly in X.

    Args:

        x, y (np.ndarray[float]) - original samples

        size (int) - number of uniform samples

        cutoff (int) - upper bound for samples (quantile, 0 to 100)

    Returns:

        x, y (np.ndarray[float]) - resampled s.t. x is uniformly distributed

    """

    if size is None:
        size = x.size

    # sort values
    sort_ind = np.argsort(x)
    xx, yy = x[sort_ind], y[sort_ind]

    # apply threshold on upper bound
    if cutoff is not None:
        threshold = np.percentile(xx, cutoff)
    else:
        threshold = xx.max()+1

    # get unique x values
    xunique = np.unique(xx)

    # filter points below threshold
    below_threshold = (xx<=threshold)
    xx, yy = xx[below_threshold], yy[below_threshold]

    # get probabilities
    x_to_count = np.vectorize(Counter(xx).get)

    # get intervals
    intervals = np.diff(xunique)
    unique_below_threshold = (xunique[:-1]<=threshold)
    intervals = intervals[unique_below_threshold]

    # assign probabilities
    x_to_cumul = np.vectorize(dict(zip(xunique[:-1][unique_below_threshold], intervals/intervals.sum())).get)
    p = x_to_cumul(xx)/x_to_count(xx)
    p[np.isnan(p)] = 0

    # generate sample
    sample_ind = np.random.choice(np.arange(xx.size), size=size, p=p)
    xu, yu = xx[sample_ind], yy[sample_ind]

    return xu, yu
