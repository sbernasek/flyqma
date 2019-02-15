import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm, multivariate_normal

from .univariate import UnivariateMixture, figure


class MixtureProperties:
    """ Extension properties for bivariate mixtures. """

    @property
    def means(self):
        return self.means_[:, 0].ravel()

    @property
    def stds(self):
        return np.sqrt(self.covariances_[:, 0].ravel())

    @property
    def supportx(self):
        max_val = np.percentile(self.values[:, 0], q=99.9)
        return np.linspace(self.values[:, 0].min(), max_val, num=100)

    @property
    def supporty(self):
        return self.supportx

    @property
    def support(self):
        xx, yy = np.meshgrid(self.supportx, self.supporty)
        return np.vstack((xx.ravel(), yy.ravel())).T.tolist()

    @property
    def extent(self):
        return np.array([self.supportx.min(), self.supportx.max(), self.supporty.max(), self.supporty.min()])

    @property
    def scale_factor(self):
        return np.exp(self.support).reshape(self.support_size[0], self.support_size[1], self.dim)

    @property
    def shape(self):
        return (self.supportx.size, self.supporty.size)

    @property
    def components(self):
        return [multivariate_normal(mean=self.means_[i], cov=self.covariances_[i]) for i in range(self.n_components)]


class BivariateMixture(UnivariateMixture,
                       MixtureProperties,
                       MixtureVisualization):
    """ Class for representing a bivariate Gaussian mixture model. """

    dim = 2

    def __getitem__(self, margin):
        """ Returns univariate mixture model for specified <margin>. """
        return self.get_marginal_mixture(margin)

    def get_marginal_mixture(self, margin):
        """ Returns univariate mixture model for specified <margin>. """

        values = self.values[:, margin]
        mu = self.means_[:, margin].reshape(-1, 1)
        cov = self.covariances_[:, margin].reshape(-1, 1)
        weights = self.weights_

        args = (mu, cov, weights)
        kwargs = dict(values=values)

        return UnivariateMixture.from_parameters(*args, **kwargs)

    def get_xmargin(self, log=True):
        pdf = np.zeros_like(self.supportx)
        for i in range(self.n_components):
            pdf += self.get_component_marginal_pdf(i, 0, True)
        if not log:
            support = np.exp(self.supportx)
            pdf /= support
        else:
            support = self.supportx
        return support, pdf

    def get_ymargin(self, log=True):
        pdf = np.zeros_like(self.supporty)
        for i in range(self.n_components):
            pdf += self.get_component_marginal_pdf(i, 1, True)
        if not log:
            support = np.exp(self.supporty)
            pdf /= support
        else:
            support = self.supporty
        return support, pdf

    def get_component_pdf(self, idx, weighted=True):

        pdf = self.components[idx].pdf(self.support).reshape(self.support_size)

        if weighted:
            pdf *= self.weights_[idx]

        return pdf

    def get_component_marginal_pdf(self, idx, margin, weighted=True):
        component = self.components[idx]
        mu = component.mean[margin]
        sigma = np.sqrt(component.cov[margin, margin])
        pdf = norm(mu, sigma).pdf(self.supportx)

        if weighted:
            pdf *= self.weights_[idx]

        return pdf


class BivariateVisualization:
    """ Visualization methods for bivariate mixture models. """

    def plot_pdf_surface(self, ax=None, contours=None, **kwargs):

        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 2))

        ax.imshow(self.pdf, extent=self.extent, **kwargs)

        if contours is not None:
            ax.contour(self.pdf, contours, extent=self.extent[[0,1,3,2]], colors='r')

        self.format_joint_ax(ax)

    def plot_margin(self, margin=0,
                    invert=False,
                    log=True,
                    pdf_color='k',
                    component_color='r',
                    ax=None,
                    **kwargs):

        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 2))

        if margin == 0:
            support, pdf = self.get_xmargin(log=log)
        else:
            support, pdf = self.get_ymargin(log=log)

        if invert:
            ax.plot(pdf, support, '-', color=pdf_color, **kwargs)
            for idx in range(self.n_components):
                mpdf = self.get_component_marginal_pdf(idx, margin, True)
                ax.plot(mpdf, support, '-', color=component_color, **kwargs)

        else:
            ax.plot(support, pdf, '-', color=pdf_color, **kwargs)
            for idx in range(self.n_components):
                mpdf = self.get_component_marginal_pdf(idx, margin, True)
                ax.plot(support, mpdf, '-', color=component_color, **kwargs)

        return ax

    def plot(self, size_ratio=4, figsize=(3, 3), contours=None, **kwargs):

        # create figure
        fig = plt.figure(figsize=figsize)
        ratios = [size_ratio/(1+size_ratio), 1/(1+size_ratio)]
        gs = GridSpec(2, 2, width_ratios=ratios, height_ratios=ratios[::-1], wspace=0, hspace=0)
        ax_joint = fig.add_subplot(gs[1, 0])
        ax_xmargin = fig.add_subplot(gs[0, 0])
        ax_ymargin = fig.add_subplot(gs[1, 1])
        ax_xmargin.axis('off')
        ax_ymargin.axis('off')

        # plot multivariate pdf surface
        self.plot_pdf_surface(ax=ax_joint, contours=contours, **kwargs)

        # plot marginal pdfs
        self.plot_margin(0, ax=ax_xmargin)
        self.plot_margin(1, invert=True, ax=ax_ymargin)

        return fig

    def scatter(self, ax, **kwargs):
        ax.scatter(*self.values.T, **kwargs)

    def format_joint_ax(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        ax.set_xlabel('ln X')
        ax.set_ylabel('neighbors <ln X>')

        ax.set_xticks(self.tick_positions)
        ax.set_yticks(self.tick_positions)
        ax.spines['left'].set_position(('outward', 1))
        ax.spines['bottom'].set_position(('outward', 1))

    def format_marginal_ax(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('ln X')

    @property
    def tick_positions(self):
        """ Tick positions. """
        roundval = lambda x: np.round(x)
        lbound = roundval(self.supportx.min())
        ubound = roundval(self.supportx.max())
        step = (ubound - lbound) // 4
        return np.arange(lbound, ubound+1, step)
