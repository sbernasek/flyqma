import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from modules.paths import Experiment
from modules.figure_settings import *

# # load example disc and layer
# disc_id = 42
# layer_id = 3

# p = '/Volumes/bernasek_t5/pnt_manuscript/microscopy/yan_clones/perturbation/slices/w-eyFLP_PntGFP_YanE833_FRT40A_PntGFP_Ubi-mRFPnls_FRT40A'

# experiment = Experiment(p)
# stack = experiment[experiment.disc_names.index(disc_id)]
# layer = stack[layer_id]

# # load example cells
# exp = df[df.experiment=='perturbation']
# disc = exp[exp.disc_id==42]
# cells = disc[disc.layer==3]


class Figure:

    def __init__(self, name='unnamed', directory='../graphics'):
        self.name = name
        self.directory = directory
        self.fig = None

    def compile(self):
        """ Compile figure. """
        pass

    @staticmethod
    def create_figure(figsize=(3, 3)):
        fig = plt.figure(figsize=figsize)
        return fig

    def add_axes(self, nrows=1, ncols=1):
        axes = self.fig.subplots(nrows=nrows, ncols=ncols)

    def save(self, fmt='pdf', dpi=300, transparent=True, rasterized=True):
        path = os.path.join(self.directory, self.name + '.{}'.format(fmt))
        kw = dict(dpi=dpi, transparent=transparent, rasterized=rasterized)
        self.fig.savefig(path, format=fmt, **kw)

    def _add_markers(self, x, y, c, **kw):
        """ Add markers to axis. """

        if len(self.fig.axes) == 0:
            ax = self.fig.subplots()
        ax = self.fig.axes[0]

        # add markers to plot
        ax.scatter(x, y, c=c, **kw)

    def format(self, **kw):
        for ax in self.fig.axes:
            self.format_axis(ax, **kw)

    def format_axis(self, ax):
        pass


class CellSelection(Figure):

    def __init__(self, layer, cells, **kw):
        Figure.__init__(self, name='selection')
        self.compile(layer, cells, **kw)

    def compile(self, layer, cells, channel='r', figsize=(3, 3)):
        """ Compile figure """

        # create figure
        self.fig = self.create_figure(figsize)
        self.add_axes()

        # add data
        self.add_image(layer, channel=channel)
        self.add_markers(cells)

    def add_image(self, layer, channel='r'):
        """ Add image to figure. """
        _ = layer.get_channel(channel).show(ax=ax, segments=False, cmap=None)
        _ = ax.axis('off')

    def add_markers(self, cells, color_by='genotype', **kw):
        """ Add markers to axis. """

        # get cell coordinates and color vector
        x, y = cells.centroid_x, cells.centroid_y

        # define colors
        c = cells[color_by]
        cmap = ListedColormap(['y', 'c', 'm'], 'indexed', 3)

        # add markers to plot
        self._add_markers(x, y, c, cmap=cmap, vmin=0, vmax=2, **kw)


class Scatterplot(Figure):
    """ Scatter points in XY plane. """

    def __init__(self, cells, x, y, name, **kw):
        Figure.__init__(self, name=name)
        self.x, self.y = x, y
        self.compile(cells, **kw)

    def compile(self, cells, figsize=(2, 2), **kw):
        """ Compile figure. """

        # create figure
        self.fig = self.create_figure(figsize)
        self.add_axes()

        # add data
        self._add_markers(cells[self.x], cells[self.y], c='k', s=1)

        # format axes
        self.format()

    def format_axis(self, ax, labelsize=7):
        _ = ax.spines['top'].set_visible(False)
        _ = ax.spines['right'].set_visible(False)
        _ = ax.set_xlabel(self.x, fontsize=labelsize)
        _ = ax.set_ylabel(self.y, fontsize=labelsize)


class BackgroundCorrelation(Scatterplot):

    def __init__(self, cells, name, figsize=(2, 2)):
        Scatterplot.__init__(self, cells, 'r', 'g', name, figsize=figsize)

    def format_axis(self, ax, labelsize=7):
        _ = ax.spines['top'].set_visible(False)
        _ = ax.spines['right'].set_visible(False)
        _ = ax.set_xticks(np.arange(0, .95, .2))
        _ = ax.set_yticks(np.arange(0, .95, .2))
        _ = ax.set_xlabel('Nuclear RFP level', fontsize=labelsize)
        _ = ax.set_ylabel('Nuclear GFP level', fontsize=labelsize)


class ComparisonFigure(Figure):

    def __init__(self, cells, y, x='genotype', mode='box', name=None, **kw):
        Figure.__init__(self, name=name)
        self.y = y
        self.x = x
        self.o = sorted(cells[x].unique())
        self.mode = mode
        self.compile(cells, mode=mode, **kw)

    def compile(self, cells, figsize=(2, 2), **kw):
        """ Compile figure. """

        # create figure
        self.fig = self.create_figure(figsize=figsize)
        self.add_axes()
        ax = self.fig.axes[0]

        # add comparison
        self._add_comparison(cells, self.y, ax, **kw)

    def _add_comparison(self, df, y, ax, **kw):
        """ Add comparison to axis. """
        if self.mode == 'box':
            sns.boxplot(ax=ax, data=df, x=self.x, order=self.o, y=y,
                    notch=True, width=0.8, **kw)

        elif self.mode == 'violin':
            sns.violinplot(ax=ax, data=df, x=self.x, order=self.o, y=y,
                    scale='width', linewidth=0.5, fliersize=0, **kw)

        elif self.mode == 'strip':
            sns.stripplot(ax=ax, data=df, x=self.x, order=self.o, y=y,
                    dodge=True, **kw)

    def format_axis(self, ax, labelsize=7):
        ax.grid(axis='y')
        ax.tick_params(axis='x', labelsize=labelsize, pad=-2)
        ax.tick_params(axis='y', labelsize=labelsize, pad=-2)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        _ = ax.spines['top'].set_visible(False)
        _ = ax.spines['right'].set_visible(False)


class ChannelComparison(ComparisonFigure):

    def __init__(self, cells, mode='violin', order=None, name=None, **kw):
        Figure.__init__(self, name=name)
        self.y0 = 'red'
        self.y1 = 'green_corrected'
        self.x = 'genotype'
        if order is None:
            order = sorted(cells[self.x].unique())
        self.o = order
        self.mode = mode
        self.compile(cells, **kw)
        plt.tight_layout()

    def compile(self, cells, figsize=(3, 1.5), grid=False, labelsize=7, **kw):
        """ Compile figure. """

        # create figure
        self.fig = self.create_figure(figsize=figsize)
        self.add_axes(ncols=2)
        ax0, ax1 = self.fig.axes

        # add comparisons
        self._add_comparison(cells, self.y0, ax0, **kw)
        self._add_comparison(cells, self.y1, ax1, **kw)
        ax0.set_ylabel('UbiRFP (a.u.)', fontsize=labelsize+1)
        ax1.set_ylabel('PntGFP (a.u.)', fontsize=labelsize+1)

        # format figure
        self.format(grid=grid, labelsize=labelsize)

    def format(self, grid=False, labelsize=7):
        """ Format figure. """

        # label axes
        ax0, ax1 = self.fig.axes

        # format axes
        for ax in self.fig.axes:
            self.format_axis(ax, labelsize=labelsize)
            if grid:
                ax.grid(axis='y')

    def format_axis(self, ax, labelsize=7, pad=-1):
        """ Format axis. """

        ax.grid('off')
        ax.tick_params(axis='x', labelsize=labelsize, pad=pad)
        ax.tick_params(axis='y', labelsize=labelsize, pad=pad)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        _ = ax.spines['top'].set_visible(False)
        _ = ax.spines['right'].set_visible(False)
        plt.rcParams['xtick.major.pad'] = tickpad
        plt.rcParams['xtick.major.pad'] = tickpad
        ax.tick_params(pad=0)

        # define labels and corresponding fill colors
        labels = {0:'−/−', 1:'−/+', 2:'+/+'}
        colors = {0:'y', 1:"c", 2:"m"}

        # format xticklabels as genotypes
        if self.mode == 'violin':
            is_poly = lambda x: x.__class__.__name__ == 'PolyCollection'
            polys = [c for c in ax.collections if is_poly(c)]
        ticklabels = []
        for i, label in enumerate(ax.get_xticklabels()):

            # get color for current label
            color = colors[int(label.get_text()[0])]
            if self.mode == 'violin':
                polys[i].set_color(color)
            else:
                ax.artists[i].set_facecolor(color)
            label.set_text(labels[int(label.get_text()[0])])
            ticklabels.append(label)
        ax.set_xlabel('')
        _ = ax.set_xticklabels(ticklabels, ha='center')

        # format ylabels
        ax.tick_params(axis='both', pad=3)


class ControlComparison(ChannelComparison):

    def __init__(self, cells, y='green_corrected', mode='violin', **kw):
        Figure.__init__(self)
        self.x = 'genotype'
        self.y = y
        self.o = sorted(cells[self.x].unique())
        self.mode = mode

        # assemble populations
        self.control = cells[cells.experiment=='control']
        self.perturbation = cells[cells.experiment=='perturbation']

        self.compile(**kw)

    def compile(self, figsize=(3, 1.5), grid=False, labelsize=7, **kw):
        """ Compile figure. """

        # create figure
        self.fig = self.create_figure(figsize=figsize)
        self.add_axes(ncols=2)
        ax0, ax1 = self.fig.axes

        # add comparisons
        self._add_comparison(self.control, self.y, ax0, **kw)
        self._add_comparison(self.perturbation, self.y, ax1, **kw)
        ax0.set_title('Control clones', fontsize=8)
        ax1.set_title('Yan clones', fontsize=8)

        # format figure
        self.format(grid=grid, labelsize=labelsize)
























