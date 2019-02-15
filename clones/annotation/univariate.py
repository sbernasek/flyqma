import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


class MixtureProperties:

    @property
    def lbound(self):
        """ Lower bound of support. """
        return np.percentile(self.values, q=0.1)

    @property
    def ubound(self):
        """ Upper bound of support. """
        return np.percentile(self.values, q=99.9)

    @property
    def bounds(self):
        """ Low and upper bounds of support. """
        return np.percentile(self.values, q=[.1, 99.9])

    @property
    def support(self):
        """ Distribution support. """
        return np.linspace(self.lbound, self.ubound, num=100)

    @property
    def num_components(self):
        """ Number of model components. """
        return self.n_components

    @property
    def means(self):
        """ Mean value of each component. """
        return self.means_.ravel()

    @property
    def stds(self):
        """ Standard deviation of each component. """
        return np.sqrt(self.covariances_.ravel())

    @property
    def log_likelihood(self):
        """ Maximized log likelihood. """
        return self.score(self.values) * self.values.size

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
    def components(self):
        """ Individual model components. """
        get_params = lambda i: (self.means_[i], np.sqrt(self.covariances_[i])
        build_component = lambda i: norm(*get_params(i)))
        return [build_component(i) for i in range(self.n_components)]

    @property
    def pdf(self):
        """ Mixture PDF. """
        pdf = np.zeros(self.support_size)
        for i in range(self.n_components):
            pdf += self.get_component_pdf(i)
        return pdf


class UnivariateMixture(GaussianMixture,
                        MixtureProperties,
                        MixtureVisualization):
    """ Class for representing a Gaussian mixture model. """

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
        model.fit(np.random.rand(5, 1))  # train on false data

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

    def get_component_pdf(self, idx, weighted=True, log=True):
        """ Returns PDF for indexed component. """

        pdf = self.components[idx].pdf(self.support).reshape(self.support_size)

        if not log:
            pdf /= self.scale_factor

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


def figure(func):
    def wrapper(*args, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(2., 1.25))
        func(*args, ax=ax, **kwargs)
    return wrapper


class MixtureVisualization:
    """ Visualization methods for mixture models. """

    @property
    def summary(self):
        """ Returns text-based summary of mixture model. """
        m = ' :: '.join(['{:0.2f}'.format(x) for x in self.means])
        s = ' :: '.join(['{:0.2f}'.format(np.sqrt(x)) for x in self.stds])
        w = ' :: '.join(['{:0.2f}'.format(x) for x in self.weights_])
        summary = 'Means: {:s}'.format(m)
        summary += '\nStds: {:s}'.format(s)
        summary += '\nWeights: {:s}'.format(w)
        summary += '\nlnL: {:0.2f}'.format(self.log_likelihood)
        return summary

    @figure
    def plot_component_pdf(self, idx,
                           weighted=True,
                           log=False,
                           ax=None,
                           **kwargs):
        """ Plots PDF for specified component. """

        # retrieve pdf for specified component
        pdf = self.get_component_pdf(idx, weighted=weighted, log=log)

        # plot component pdf
        if log:
            ax.plot(self.support, pdf, **kwargs)
        else:
            ax.plot(self.scale_factor, pdf, **kwargs)

        self.format_ax(ax, log=log)

    @figure
    def plot_pdf(self, log=False, ax=None, **kwargs):
        """ Plots overall PDF for mixture model. """

        if log:
            ax.plot(self.support, self.pdf*self.scale_factor, **kwargs)
        else:
            ax.plot(self.scale_factor, self.pdf, **kwargs)

        self.format_ax(ax, log=log)

    @figure
    def plot(self, log=False, ax=None,
             pdf_color='k', component_color='r', **kwargs):
        """ Plots PDF for mixture model as well as each weighted component. """

        self.plot_pdf(log=log, ax=ax, color=pdf_color)
        for i in range(self.n_components):
            self.plot_component_pdf(i, log=log, ax=ax, color=component_color)

    @figure
    def plot_data(self, log=False, ax=None, **kwargs):
        """ Plots PDF for mixture model as well as each weighted component. """

        if log:
            data = self.values
        else:
            data = np.exp(self.values)

        bins = np.linspace(self.support.min(), self.support.max(), num=50)
        _ = ax.hist(data, bins=bins, density=True, **kwargs)

    def format_ax(self, ax, log=False):
        if log:
            ax.set_xlim(self.support.min(), self.support.max())
        else:
            ax.set_xlim(0, self.scale_factor.max())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

