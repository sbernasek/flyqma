import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ...visualization import *


def figure(func):
    """ Decorator for creating axis. """
    def wrapper(*args, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(2., 1.25))
        func(*args, ax=ax, **kwargs)
    return wrapper

def surface_figure(func):
    """ Decorator for creating joint axis. """
    def wrapper(*args, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(2., 2.))
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
                           log=True,
                           ax=None,
                           **kwargs):
        """ Plots PDF for specified component. """

        # retrieve pdf for specified component
        pdf = self.get_component_pdf(idx, weighted=weighted)

        # plot component pdf
        if log:
            ax.plot(self.support, pdf, **kwargs)
        else:
            ax.plot(self.scale_factor, pdf/self.scale_factor, **kwargs)

        self.format_ax(ax, log=log)

    @figure
    def plot_pdf(self, log=True, ax=None, **kwargs):
        """ Plots overall PDF for mixture model. """

        if log:
            ax.plot(self.support, self.pdf, **kwargs)
        else:
            ax.plot(self.scale_factor, self.pdf/self.scale_factor, **kwargs)

        self.format_ax(ax, log=log)

    @figure
    def plot(self, log=True, ax=None,
             pdf_color='k', component_color='r', **kwargs):
        """ Plots PDF for mixture model as well as each weighted component. """
        self.plot_pdf(log=log, ax=ax, color=pdf_color)
        for i in range(self.n_components):
            self.plot_component_pdf(i, log=log, ax=ax, color=component_color)

    @figure
    def plot_data(self, log=True, ax=None, **kwargs):
        """ Plot binned values. """

        if log:
            data = self.values
        else:
            data = np.exp(self.values)

        bins = np.linspace(self.support.min(), self.support.max(), num=50)
        _ = ax.hist(data, bins=bins, density=True, **kwargs)

    def format_ax(self, ax, log=True):
        if log:
            ax.set_xlim(self.support.min(), self.support.max())
        else:
            ax.set_xlim(0, self.scale_factor.max())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


class BivariateVisualization:
    """ Visualization methods for bivariate mixture models. """

    @surface_figure
    def plot_pdf_surface(self, ax=None, contours=None, **kwargs):

        ax.imshow(self.pdf, extent=self.extent, **kwargs)

        if contours is not None:
            ax.contour(self.pdf, contours, extent=self.extent[[0,1,3,2]], colors='r')

        self.format_joint_ax(ax)

    @figure
    def plot_margin(self, margin=0,
                    invert=False,
                    log=True,
                    pdf_color='k',
                    pdf_linestyle='--',
                    pdf_lw=0.5,
                    component_color='r',
                    component_lw=0.5,
                    min_density=0.01,
                    ax=None):

        # extract x/y margins
        if margin == 0:
            support, pdf = self.get_xmargin(log=log)
        else:
            support, pdf = self.get_ymargin(log=log)

        # define colors for component lines
        if type(component_color) == str:
            component_color *= self.n_components

        # define parameters
        above = (pdf >= min_density)
        pdf_kw = dict(color=pdf_color, lw=pdf_lw, linestyle=pdf_linestyle)
        comp_kw = dict(lw=component_lw)

        # invert x/y axes (for vertical margin)
        if invert:
            ax.plot(pdf[above], support[above], '-', **pdf_kw)
            for i in range(self.n_components):
                mpdf = self.get_component_marginal_pdf(i, margin, True)
                ind = (mpdf >= min_density)
                ax.plot(mpdf[ind], support[ind], color=component_color[i], **comp_kw)
        else:
            ax.plot(support[above], pdf[above], '-', **pdf_kw)
            for i in range(self.n_components):
                mpdf = self.get_component_marginal_pdf(i, margin, True)
                ind = (mpdf >= min_density)
                ax.plot(support[ind], mpdf[ind], color=component_color[i], **comp_kw)

        return ax

    def visualize(self,
                  size_ratio=4,
                  figsize=(2, 2),
                  contours=None,
                  **kwargs):
        """ Visualize joint and marginal distributions. """

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

        ax_xmargin.set_xlim(self.lbound, self.ubound)
        ax_ymargin.set_ylim(self.lbound, self.ubound)

        return fig

    @figure
    def plot_data(self, ax=None, **kwargs):
        """ Scatter datapoints on <ax>. """
        ax.scatter(*self.values.T, **kwargs)

    def format_joint_ax(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        ax.set_xlabel('ln X')
        ax.set_ylabel('<ln X> among neighbors')

        ax.set_xticks(self.tick_positions)
        ax.set_yticks(self.tick_positions)
        ax.spines['left'].set_position(('outward', 2))
        ax.spines['bottom'].set_position(('outward', 2))

    def format_marginal_ax(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('ln X')

    @property
    def tick_positions(self):
        """ Tick positions. """
        lbound = np.ceil(self.supportx.min())
        ubound = np.floor(self.supportx.max())
        step = (ubound - lbound) // 4
        return np.arange(lbound, ubound+1, step)
