import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

from ...visualization import *


class MixtureVisualization:
    """ Methods for visualizing a mixture-model based classifier. """

    @property
    def label_colors(self):
        """ RGB color for each class label. """
        component_to_label = np.vectorize(self.component_to_label.get)
        labels = component_to_label(np.arange(self.num_components))
        label_colors = self.cmap(labels)[:, :-1]
        return label_colors

    @property
    def support(self):
        """ Model support. """
        return self.model.support

    @property
    def esupport(self):
        """ Empirical support vector (sorted values). """
        return np.sort(self.values, axis=0)

    @property
    def epdf(self):
        """ Empirical PDF over support. """
        num_bins = self.num_samples // 25
        bins = np.linspace(self.support.min(), self.support.max(), num_bins)
        counts, edges = np.histogram(self.values, bins=bins, normed=True)
        bin_centers = [(edges[i]+edges[i+1])/2. for i in range(len(edges)-1)]
        return edges[:-1], counts

    @property
    def ecdf(self):
        """ Empirical CDF over support. """
        return np.linspace(0, 1, len(self.values), endpoint=False)

    @property
    def pdf(self):
        """ Model PDF over support. """
        return self.model.pdf

    @property
    def component_pdfs(self):
        """ Weighted component PDFs over support. """
        return self.model.component_pdfs

    @property
    def component_cdfs(self):
        """ Returns weighted CDF of each component over support. """
        cdfs = np.vstack([x.cdf(self.support) for x in self.model.components])
        cdfs *= self.model.weights_.reshape(-1, 1)
        return cdfs

    @property
    def support_labels(self):
        """ Labels for support vector. """
        return self.classifier(self.support.reshape(-1, 1))

    @default_figure
    def plot_pdf(self,
                  density=1000,
                  alpha=0.5,
                  xmin=None,
                  xmax=None,
                  ymin=None,
                  ymax=None,
                  ax=None):
        """
        Plot model density function, colored by output label.
        """

        # plot model pdf segments, colored by output label
        support_labels = self.support_labels
        breakpoints = [0]+list(np.diff(support_labels).nonzero()[0]+1)+[None]
        for i, bp in enumerate(breakpoints[:-1]):
            indices = slice(bp, breakpoints[i+1])
            segment_support = self.support[indices]
            segment_pdf = self.pdf[indices]
            segment_labels = support_labels[indices]
            segment_color = self.cmap(segment_labels)
            ax.fill_between(segment_support, segment_pdf, color=segment_color)

        # plot overall model pdf
        ax.plot(self.support, self.pdf, '-', c='k', lw=2)

        # format axis
        if ymax is None:
            maxima = self.pdf[argrelextrema(self.pdf, np.greater)]
            ymax = 2.5*np.product(maxima)**(1/maxima.size)

        ax.set_xlim(self.model.lbound, self.model.ubound)
        ax.set_ylim(0, ymax)
        ax.set_xlabel('Values', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)

    @default_figure
    def plot_pdfs(self,
                  empirical=False,
                  line=True,
                  fill=True,
                  density=1000,
                  alpha=0.5,
                  cmap=None,
                  vmin=-1,
                  xmax=None,
                  ymax=None,
                  ax=None):
        """
        Plot density function for each distribution, colored by output label.

        Args:

            ax (matplotlib.axes.AxesSubplot) - if None, create figure

            empirical (bool) - if True, include empirical PDF

        """

        # define colormap
        if cmap is not None:
            colormap = self.build_colormap(cmap, vmin)
        else:
            colormap = self.cmap

        # plot empirical pdf
        if empirical:
            ax.step(*self.epdf, where='post', color='r', linewidth=1)

        # plot individual component pdfs
        for i, pdf in enumerate(self.component_pdfs):
            color = colormap(self.component_to_label[i])
            if line:
                ax.plot(self.support, pdf, color=color, alpha=alpha, lw=1.)
            if fill:
                ax.fill_between(self.support, pdf, facecolors=color, alpha=alpha, linewidth=1., rasterized=True)

        # plot model pdf
        ax.plot(self.support, self.pdf, '--', c='k', lw=1)

        # format axis
        if ymax is None:
            maxima = self.pdf[argrelextrema(self.pdf, np.greater)]
            ymax = 2.5*np.product(maxima)**(1/maxima.size)

        ax.set_ylim(0, ymax)
        ax.set_xlim(self.model.lbound, self.model.ubound)
        ax.set_xlabel('Values', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)

    @default_figure
    def plot_cdfs(self,
                  log=True,
                  cmap=plt.cm.Greys,
                  ax=None,
                  **kwargs):
        """
        Plot component cumulative distribution functions as stackplot.
        """

        def to_linear(support):
            """ Convert support to linear basis. """
            return np.exp(support)

        # log transform data
        support, esupport = self.support, self.esupport
        if not log:
            support, esupport = to_linear(support), to_linear(esupport)

        # get component CDFs
        component_cdfs = self.component_cdfs

        # plot weighted CDF for each component
        means = self.means
        values = self.values
        norm = Normalize(vmin=self.values.min(), vmax=self.values.max())
        order = np.argsort(means)
        colors = cmap(norm(means[order]))
        ax.stackplot(support, component_cdfs[order], colors=colors, **kwargs)

        # plot empirical CDF (data)
        ax.plot(esupport, self.ecdf, '-r', lw=1.)

        # plot mixture CDF
        ax.plot(support, component_cdfs.sum(axis=0), '--k', lw=1)


class BivariateMixtureVisualization:

    @property
    def support(self):
        """ Model support. """
        return self.model.supportx

    @property
    def esupport(self):
        """ Empirical support vector (sorted values). """
        return np.sort(self.values[:, 0])

    @property
    def pdf(self):
        """ Model PDF over support. """
        return self.model[0].pdf

    @property
    def component_pdfs(self):
        """ Weighted component PDFs over support. """
        return self.model[0].component_pdfs

    @property
    def component_cdfs(self):
        """ Returns weighted CDF of each component over support. """
        model = self.model[0]
        cdfs = np.vstack([x.cdf(self.support) for x in model.components])
        cdfs *= model.weights_.reshape(-1, 1)
        return cdfs

    @property
    def support_labels(self):
        """ Labels for support vector (over x margin). """
        margin = self.marginalize(0)
        return margin.classifier(self.support.reshape(-1, 1))

    @joint_figure
    def plot_bivariate_pdf(self, fig, bg='w', **kwargs):
        """ Plot bivariate PDF, with each cluster shaded by its label. """

        pdfs = self.model.component_pdfs

        # plot each bivariate pdf, colored by label
        norm = Normalize(pdfs.min(), pdfs.max())
        fig.ax_joint.set_facecolor(bg)
        for idx, pdf in enumerate(pdfs):
            ccm = build_transparent_cmap(self.label_colors[idx], bg=bg)
            fig.ax_joint.imshow(norm(pdf), cmap=ccm, extent=self.model.extent)

        # plot marginal pdfs
        self.model.plot_margin(0, ax=fig.ax_xmargin, component_color=self.label_colors)
        self.model.plot_margin(1, invert=True, ax=fig.ax_ymargin, component_color=self.label_colors)

        fig.ax_xmargin.set_xlim(self.model.lbound, self.model.ubound)
        fig.ax_ymargin.set_ylim(self.model.lbound, self.model.ubound)

        return fig

    @joint_figure
    def plot_bivariate_data(self, fig, bg='w', **kwargs):
        """ Plot bivariate data, with each sample shaded by its label. """

        # define label colors
        marker_labels = self.classifier(self.model.values)
        marker_colors = self.cmap(marker_labels)[:, :-1]

        # plot each bivariate pdf, colored by label
        fig.ax_joint.set_facecolor(bg)
        self.model.plot_data(ax=fig.ax_joint, c=marker_colors, **kwargs)
        fig.ax_joint.set_xlim(self.model.lbound, self.model.ubound)
        fig.ax_joint.set_ylim(self.model.lbound, self.model.ubound)
        fig.ax_joint.invert_yaxis()
        fig.ax_joint.set_yticks(fig.ax_joint.get_xticks())

        # plot marginal pdfs
        self.model.plot_margin(0, ax=fig.ax_xmargin, component_color=self.label_colors)
        self.model.plot_margin(1, invert=True, ax=fig.ax_ymargin, component_color=self.label_colors)
        fig.ax_xmargin.set_xlim(self.model.lbound, self.model.ubound)
        fig.ax_ymargin.set_ylim(self.model.lbound, self.model.ubound)

        return fig

    @joint_figure
    def plot_phase_space(self, fig, bg='w', **kwargs):
        """ Plot phase space, with each region shaded by its label. """

        labels = self.classifier(self.model.support)
        labels = labels.reshape(self.model.support_size)

        l, u = self.model.lbound, self.model.ubound
        extent = (l, u, u, l)
        fig.ax_joint.imshow(self.cmap(labels), extent=extent)

        # plot marginal pdfs
        self.model.plot_margin(0, ax=fig.ax_xmargin, component_color=self.label_colors)
        self.model.plot_margin(1, invert=True, ax=fig.ax_ymargin, component_color=self.label_colors)

        fig.ax_xmargin.set_xlim(self.model.lbound, self.model.ubound)
        fig.ax_ymargin.set_ylim(self.model.lbound, self.model.ubound)

        return fig
