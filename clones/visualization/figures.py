from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from .settings import *


class Figure:
    """
    Base class for figures providing some common methods.

    Attributes:

        name (str) - figure name

        directory (str) - default path for saving figure

        fig (matplotlib.figure.Figure)

        axes (matplotlib.axes.AxesSubplots)

    """

    def __init__(self, name='unnamed', directory='../graphics'):
        self.name = name
        self.directory = directory
        self.fig = None

    @staticmethod
    def create_figure(figsize=(3, 3)):
        """
        Create blank figure.

        Args:

            figsize (tuple) - figure dimensions
        """
        fig = plt.figure(figsize=figsize)
        return fig

    def add_axes(self, nrows=1, ncols=1):
        """
        Add axes to figure.

        Args:

            nrows, ncols (int) - number of rows and columns

        """
        self.axes = self.fig.subplots(nrows=nrows, ncols=ncols)

    def save(self, **kwargs):
        """
        Save figure to file.

        Keyword Arguments:

            fmt (str) - file format, eg 'pdf'

            dpi (int) - resolution

            transparent (bool) - if True, remove background

            rasterized (bool) - if True, rasterize figure data

            addtl kwargs: keyword arguments for plt.savefig

        """

        self._save(self.fig, self.name, self.directory, **kwargs)

    @staticmethod
    def _save(fig,
             name,
             dirpath='./',
             fmt='pdf',
             dpi=300,
             transparent=True,
             rasterized=True,
             **kwargs):
        """
        Save figure to file.

        Args:

            fig (matplotlib.figures.Figure) - figure to be saved

            name (str) - file name without format extension

            dirpath (str) - directory in which to save file

            fmt (str) - file format, eg 'pdf'

            dpi (int) - resolution

            transparent (bool) - if True, remove background

            rasterized (bool) - if True, rasterize figure data

            kwargs: keyword arguments for plt.savefig

        """
        path = join(dirpath, name+'.{}'.format(fmt))
        kw = dict(dpi=dpi, transparent=transparent, rasterized=rasterized)
        fig.savefig(path, format=fmt, **kwargs)

    def _add_markers(self, x, y, c, **kwargs):
        """
        Add markers to axis.

        Args:

            x, y (array like) - marker x and y positions

            c (array like) - marker colors

            kwargs: keyword arguments for matplotlib.pyplot.scatter

        """

        if len(self.fig.axes) == 0:
            ax = self.fig.subplots()
        ax = self.fig.axes[0]

        # add markers to plot
        ax.scatter(x, y, c=c, **kwargs)

    def format(self, **kwargs):
        """ Format all figure axes. """
        for ax in self.fig.axes:
            self.format_axis(ax, **kwargs)

    def format_axis(self, ax):
        """ Format individual axis. """
        pass


class CellSelection(Figure):
    """
    Visualize cell selection by overlaying cell position markers on top of an image of a single RGB layer.

    Inherited attributes:

        name (str) - figure name ('selection')

        directory (str) - default path for saving figure

        fig (matplotlib.figure.Figure)

        axes (matplotlib.axes.AxesSubplots)

    """

    def __init__(self, layer, data, channel='r', **kwargs):
        """
        Instantiate cell selection figure.

        Args:

            layer (Layer) - RGB image layer

            data (pd.DataFrame) - selected cell measurement data

            channel (str) - color channel to be added

            kwargs: keyword arguments for render

        """
        Figure.__init__(self, name='selection')
        self.render(layer, data, **kwargs)

    def render(self, layer, data, channel='r', figsize=(3, 3)):
        """
        Render figure.

        Args:

            layer (Layer) - RGB image layer

            data (pd.DataFrame) - selected cell measurement data

            channel (str) - color channel to be added

            figsize (tuple) - figure dimensions

        """

        # create figure
        self.fig = self.create_figure(figsize)
        self.add_axes()

        # add image
        self.add_image(layer, channel=channel)

        # add cell position markers
        self.add_markers(data)

    def add_image(self, layer, channel='r'):
        """
        Add scalar image to figure.

        Args:

            layer (Layer) - RGB image layer

            channel (str) - color channel to be added

        """
        _ = layer.get_channel(channel).show(ax=ax, segments=False, cmap=None)
        _ = ax.axis('off')

    def add_markers(self, data, color_by='genotype', xykey=None, **kwargs):
        """
        Add cell position markers to axis.

        Args:

            data (pd.DataFrame) - selected cell measurement data

            color_by (str) - cell measurement attribute used to color markers

            xykey (list) - attribute keys for cell x/y positions

            kwargs: keyword arguments for markers

        """

        if xykey is None:
            xykey = ['centroid_x', 'centroid_y']

        # get cell coordinates and color vector
        x, y = data[xykey].values.T

        # get color vector and colormap
        c = data[color_by]
        cmap = ListedColormap(['y', 'c', 'm'], 'indexed', 3)

        # add markers to plot
        self._add_markers(x, y, c, cmap=cmap, vmin=0, vmax=2, **kwargs)


class Scatterplot(Figure):
    """
    Scatter points in XY plane.

    Attributes:

        xvar, yvar (str) - cell measurement features to be scattered

    Inherited attributes:

        name (str) - figure name

        directory (str) - default path for saving figure

        fig (matplotlib.figure.Figure)

        axes (matplotlib.axes.AxesSubplots)

    """

    def __init__(self, data, xvar, yvar, name, **kwargs):
        """
        Instantiate scatter plot.

        Args:

            data (pd.DataFrame) - selected cell measurement data

            xvar, yvar (str) - cell measurement features to be scattered

            name (str) - figure name

            kwargs: keyword arguments for

        """
        Figure.__init__(self, name=name)
        self.xvar, self.yvar = xvar, yvar
        self.render(data, **kwargs)

    def render(self, data, figsize=(2, 2)):
        """
        Render figure.

        Args:

            data (pd.DataFrame) - selected cell measurement data

            figsize (tuple) - figure dimensions

        """

        # create figure
        self.fig = self.create_figure(figsize)
        self.add_axes()

        # add data
        self._add_markers(data[self.xvar], data[self.yvar], c='k', s=1)

        # format axes
        self.format()

    def format_axis(self, ax):
        """
        Format axis.

        Args:

            ax (matplotlib.axes.AxesSubplot)

        """
        _ = ax.spines['top'].set_visible(False)
        _ = ax.spines['right'].set_visible(False)
        _ = ax.set_xlabel(self.x)
        _ = ax.set_ylabel(self.y)


class BackgroundCorrelation(Scatterplot):
    """
    Plot correlated expression between red and green fluorescence channels.

    Inherited attributes:

        xvar, yvar (str) - cell measurement features to be scattered

        name (str) - figure name

        directory (str) - default path for saving figure

        fig (matplotlib.figure.Figure)

        axes (matplotlib.axes.AxesSubplots)

    """

    def __init__(self, data, name, figsize=(2, 2)):
        """
        Instantiate background correlation plot.

        Args:

            data (pd.DataFrame) - selected cell measurement data

            name (str) - figure name

            figsize (tuple) - figure size

        """
        Scatterplot.__init__(self, data, 'r', 'g', name, figsize=figsize)

    def format_axis(self, ax):
        """
        Format axis.

        Args:

            ax (matplotlib.axes.AxesSubplot)

        """
        _ = ax.spines['top'].set_visible(False)
        _ = ax.spines['right'].set_visible(False)
        _ = ax.set_xticks(np.arange(0, .95, .2))
        _ = ax.set_yticks(np.arange(0, .95, .2))
        _ = ax.set_xlabel('Nuclear RFP level')
        _ = ax.set_ylabel('Nuclear GFP level')
