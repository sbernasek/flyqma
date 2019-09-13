import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from .visualization import MixtureVisualization


class MixtureIO:

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["values"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.values = None


class MixtureProperties:
    """ Properties for guassian mixture models. """

    @property
    def means(self):
        """ Mean value of each component. """
        return self.means_[:, 0].ravel()

    @property
    def stds(self):
        """ Standard deviation of each component. """
        return np.sqrt(self.covariances_[:, 0].ravel())

    @property
    def lbound(self):
        """ Lower bound of support. """
        return np.percentile(self.values[:, 0], q=0.1)

    @property
    def ubound(self):
        """ Upper bound of support. """
        return np.percentile(self.values[:, 0], q=99.9)

    @property
    def bounds(self):
        """ Low and upper bounds of support. """
        return np.percentile(self.values[:, 0], q=[.1, 99.9])

    @property
    def support(self):
        """ Distribution support. """
        return np.linspace(self.lbound, self.ubound, num=100)

    @property
    def num_components(self):
        """ Number of model components. """
        return self.n_components

    @property
    def log_likelihood(self):
        """ Maximized log likelihood. """
        return self.score(self.values) * self.num_samples

    @property
    def BIC(self):
        """ BIC score. """
        return self.bic(self.values)

    @property
    def AIC(self):
        """ AIC score. """
        return self.aic(self.values)

    @property
    def scale_factor(self):
        """ Scaling factor for log-transformed support. """
        return np.exp(self.support)

    @property
    def support_size(self):
        """ Size of support. """
        return self.support.size

    @property
    def num_samples(self):
        """ Number of samples. """
        return self.values.shape[0]

    @property
    def components(self):
        """ Individual model components. """
        get_params = lambda i: (self.means_[i], np.sqrt(self.covariances_[i]))
        build_component = lambda i: norm(*get_params(i))
        return [build_component(i) for i in range(self.n_components)]

    @property
    def pdf(self):
        """ Gaussian Mixture PDF. """
        pdf = np.zeros(self.support_size)
        for i in range(self.n_components):
            pdf += self.get_component_pdf(i)
        return pdf

    @property
    def component_pdfs(self):
        """ Returns stacked array of component PDFs. """
        return self._component_pdfs()

    def _component_pdfs(self, weighted=True):
        """ Returns stacked array of component PDFs over support. """
        build_pdf = lambda idx: self.get_component_pdf(idx, weighted)
        return np.stack([build_pdf(idx) for idx in range(self.n_components)])


class UnivariateMixture(GaussianMixture,
                        MixtureIO,
                        MixtureProperties,
                        MixtureVisualization):
    """
    Univariate Gaussian mixture model.

    Attributes:

        values (array like) - values to which model was fit

    Inherited attributes:

        See sklearn.mixture.GaussianMixture

    """

    dim = 1

    def __init__(self, *args, values=None, **kwargs):
        """ Instantiate Gaussian mixture model. """
        self.values = values
        GaussianMixture.__init__(self, *args, **kwargs)

    def __repr__(self):
        """ Displays summary of mixture components. """
        return self.summary

    @classmethod
    def from_logsample(cls, sample,
                    n=3,
                    max_iter=10000,
                    tol=1e-8,
                    covariance_type='diag',
                    n_init=10):
        """ Instantiate from log-transformed sample. """

        if len(sample.shape) != cls.dim + 1:
            sample = sample.reshape(-1, cls.dim)

        model = cls(n, values=sample,
                    covariance_type=covariance_type,
                    tol=tol,
                    n_init=n_init,
                    max_iter=max_iter)

        model = model.fit(sample)

        return model

    @classmethod
    def from_sample(cls, sample, n, **kwargs):
        """ Instantiate from log-normally distributed sample. """
        return cls.from_logsample(np.log(sample), n=n, **kwargs)

    @classmethod
    def from_parameters(cls, mu, sigma, weights=None, values=None, **kwargs):
        """ Instantiate model from parameter vectors. """

        model = cls(len(mu), values=values, covariance_type='diag', **kwargs)
        model.fit(np.random.rand(50, 1))  # train on false data

        if type(mu) == list:
            mu = np.array(mu).reshape(-1, 1)

        if type(sigma) == list:
            sigma = np.array(sigma).reshape(-1, 1)

        elif type(sigma) in (int, float, np.float64, np.int64):
            sigma = np.ones((len(mu), 1), dtype=float) * sigma

        if weights is None:
            weights = np.ones(len(mu), dtype=float) / len(mu)
        elif type(weights) == list:
            weights = np.array(weights)

        model.weights_ = weights
        model.means_ = mu
        model.covariances_ = sigma
        precisions = np.linalg.inv(np.diag(sigma.ravel()))
        model.precisions_ = np.diag(precisions).reshape(-1, 1)
        cholesky_precision = np.linalg.cholesky(precisions)
        model.precisions_cholesky_ = np.diag(cholesky_precision).reshape(-1, 1)
        model.lower_bound_ = model.score(model.values)

        if values is None:
            model.values = model.logsample(1000)

        return model

    def logsample(self, N):
        """ Returns <N> samples of log-transformed variable. """
        logsamples, labels = GaussianMixture.sample(self, N)
        return logsamples

    def sample(self, N):
        """ Returns <N> samples of variable. """
        return np.exp(self.logsample(N))

    def sample_component(self, component_idx, N):
        """ Returns <N> log-transformed samples from indexed component. """
        return self.components[component_idx].rvs(N)

    def multi_logsample(self, N, m=10):
        """
        Returns <N> log-transformed samples as well as <N> log-transformed samples averaged over <m> other samples from the same component.
        """

        component_idxs = np.random.choice(np.arange(self.n_components), size=N, p=self.weights_)
        N_per_component = Counter(component_idxs)

        data = {}
        for component_idx, num_samples in N_per_component.items():
            subsample = self.sample_component(component_idx, num_samples)
            neighbor_idxs = np.random.randint(0, num_samples, size=(num_samples, m))
            neighbor_average = subsample[neighbor_idxs].mean(axis=1)
            data[component_idx] = (subsample, neighbor_average)
        data = np.hstack(data.values())

        return data.T

    def multi_sample(self, N, m=10):
        """
        Returns <N> samples as well as <N> samples averaged over <m> other samples from the same component.
        """
        return np.exp(self.multi_logsample(N, m))

    def get_component_pdf(self, idx, weighted=True):
        """ Returns PDF for indexed component. """

        pdf = self.components[idx].pdf(self.support).reshape(self.support_size)

        if weighted:
            pdf *= self.weights_[idx]

        return pdf

    def estimate_required_samples(self, SNR=5.):
        """
        Returns minimum number of averaged samples required to achieve the specified signal to noise (SNR) ratio.
        """

        yy, xx = np.meshgrid(*(np.arange(self.n_components),)*2)
        delta_mu = np.abs(self.means[xx]-self.means[yy])
        max_sigma = np.stack((self.stds[xx], self.stds[yy])).max(axis=0)

        idx = np.triu_indices(self.n_components, k=1)
        min_num_samples = (SNR*(max_sigma[idx]/delta_mu[idx])**2).max()

        return min_num_samples
