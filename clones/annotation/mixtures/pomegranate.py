import numpy as np
import matplotlib.pyplot as plt
import pomegranate as pmg

from .mixtures import UnivariateMixture


class PomegranateUnivariateMixture(UnivariateMixture):

    def __init__(self, model, values=None):
        self.model = model
        self.values = values

    @property
    def means(self):
        return [x.parameters[0] for x in self.components]

    @property
    def stds(self):
        return [x.parameters[1] for x in self.components]

    @property
    def log_likelihood(self):
        return self.model.log_probability(self.values).sum()

    @property
    def components(self):
        return self.model.distributions

    @property
    def n_components(self):
        return len(self.components)

    @property
    def weights_(self):
        return np.exp(self.model.weights)

    @classmethod
    def from_parameters(cls, mu, sigma, weights=None):

        n_components = len(mu)

        if weights is None:
            weights = np.ones(n_components, dtype=float)/n_components

        if type(sigma) not in (np.ndarray, list, tuple):
            sigma = [sigma for _ in range(n_components)]

        build_dist = lambda m, s: pmg.distributions.NormalDistribution(m, s)
        components = [build_dist(m, s) for m, s in zip(mu, sigma)]
        model = pmg.GeneralMixtureModel(components, weights=weights)

        return cls(model, np.array(model.sample(1000)))

    @classmethod
    def from_logsample(cls, sample, n=3,
                    max_iter=1e8,
                    n_init=100,
                    stop_threshold=0.001):
        if len(sample.shape) != cls.dim + 1:
            sample = sample.reshape(-1, cls.dim)
        model = cls._fit_normal_mixture(sample, n=n, max_iter=max_iter, n_init=n_init, stop_threshold=stop_threshold)
        return cls(model, sample)

    @classmethod
    def from_sample(cls, sample, **kwargs):
        return cls.from_logsample(np.log(sample), **kwargs)

    @staticmethod
    def _fit_mixture(values, model='lognormal', n=3, n_init=100, max_iter=1e8, stop_threshold=0.001):
        """ Fit model with <n> components to sampled <values>. """

        if model == 'lognormal':
            model = pmg.distributions.LogNormalDistribution
        elif model == 'normal':
            model = pmg.distributions.NormalDistribution

        args = (
            model,
            n,
            values)

        kwargs = dict(
            n_init=n_init,
            init='kmeans++',
            max_kmeans_iterations=100,
            stop_threshold=stop_threshold,
            max_iterations=max_iter,
        )

        return pmg.GeneralMixtureModel.from_samples(*args, **kwargs)

    @classmethod
    def _fit_lognormal_mixture(cls, values, n=3, **kwargs):
        """ Fit model with <n> lognormal components to sampled <values>. """
        return cls._fit_mixture(values, 'lognormal', n=n, **kwargs)

    @classmethod
    def _fit_normal_mixture(cls, values, n=3, **kwargs):
        """ Fit model with <n> normal components to sampled <values>. """
        return cls._fit_mixture(values, 'normal', n=n, **kwargs)

    def logsample(self, N):
        return np.array(self.model.sample(N)).reshape(-1, 1)

    def sample(self, N):
        return np.exp(self.logsample(N))

    def sample_component(self, component_idx, N):
        """ Returns <N> samples from indexed component. """
        return self.components[component_idx].sample(N)

    def get_component_pdf(self, idx, weighted=True, log=False):

        pdf = self.components[idx].probability(self.support).reshape(self.shape)

        if not log:
            pdf /= self.scale_factor

        if weighted:
            pdf *= self.weights_[idx]

        return pdf
