import numpy as np
from pomegranate import LogNormalDistribution

from .bayesian import BayesianProperties


class LognormalModel(BayesianProperties):
    """
    Standalone Bayesian lognormal model.

    Attributes:

        values (array like) - sample

        model (pomegranate.LogNormalDistribution) - frozen model

        fig (matplotlib.figures.Figure) - histogram figure

    """

    def __init__(self, values, weights=None, crop=True):
        """
        Fit a lognormal distribution to an array of values.

        Args:

            values (np.ndarray[float]) - 1D array of measured values

            weights (np.ndarray[float]) - 1D array of sample weights (optional)

            crop (bool) - if True, crop values to [0.1, 99.9] percentiles

        """

        # store parameters
        self.values = values
        self.sample_weights = weights
        self.crop = crop
        self.support = np.sort(values)
        self.fig = None

        # fit model
        self.model = self._fit(self.values, weights=weights, crop=crop)

    @property
    def num_parameters(self):
        """ Number of model parameters. """
        return len(self.model.parameters)

    @staticmethod
    def _fit(values, weights=None, crop=True):
        """
        Fit log-normal mixture model using likelihood maximization.

        Args:

            values (np.ndarray[float]) - 1D array of values

            weights (np.ndarray[float]) - 1D array of sample weights

            crop (bool) - if True, crop values to [0.1, 99.9] percentiles

        Returns:

            model (pomegranate.LogNormalDistribution)

        """
        if crop:
            select_between = lambda x, b: x[(x>=b[0]) * (x<=b[1])]
            bounds = np.percentile(values, q=[0.1, 99.9])
            values = select_between(values, bounds)

        return LogNormalDistribution.from_samples(values.reshape(-1, 1), weights=weights)

