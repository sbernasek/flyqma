from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec

from ..utilities.io import IO


class LayerVisualization:
    """
    Object for visualizing a single layer.

    Attributes:
    path (str) - layer path
    axes (array like) - axes for blue/red/green color channels
    """

    def __init__(self, layer, axes):
        """
        Instantiate layer visualization.

        Args:
        layer (Layer) - image layer
        axes (array like) - axes for blue/red/green color channels
        """

        # set selection path
        layer.make_subdir('selection')
        self.path = layer.subdirs['selection']

        # set axes
        self.axes = axes

        # render images
        self.render_images(layer)

    def render_images(self, layer, cmap=None):
        """ Add blue, green, and red channels of layer to axes. """

        # if cmap is None, use natural colors
        if cmap is None:
            b, r, g = 'brg'
        else:
            b, r, g = (cmap,)*3

        # visualize layers
        ax0, ax1, ax2 = self.axes
        _ = layer.get_channel('b').show(segments=False, ax=ax0, cmap=b)
        _ = layer.get_channel('r').show(segments=False, ax=ax1, cmap=r)
        _ = layer.get_channel('g').show(segments=False, ax=ax2, cmap=g)
        for ax in self.axes:
            ax.set_aspect(1)

        # add layer number
        text = ' {:d}'.format(layer._id)
        ax0.text(0, 0, text, fontsize=14, color='y', va='top')

    def add_marker(self, x, y, color='k', markersize=10):
        """ Add marker to layer images. """
        for ax in self.axes:
            ax.plot(x, y, '.', color=color, markersize=markersize, zorder=2)

    def remove_marker(self):
        """ Remove marker from layer images. """
        for ax in self.axes:
            ax.lines[-1].remove()

    def update_marker(self, color, markersize, ind=-1):
        """ Update size and color of last added marker. """
        for ax in self.axes:
            if len(ax.lines) > 0:
                ax.lines[ind].set_color(color)
                ax.lines[ind].set_markersize(markersize)

    def clear_markers(self):
        """" Remove all markers. """
        while len(self.axes[0].lines) > 0:
            self.remove_marker()

    def add_polygon(self):
        """ Add polygon to each image. """
        for ax in self.axes:
            poly = Polygon(self.pts,
                       ec=(1,1,0,1), lw=1,
                       fc=(1,1,1,0.2), fill=False,
                       zorder=1, closed=True)
            ax.add_patch(poly)

    def remove_polygon(self):
        """ Remove polygon from each image. """
        for ax in self.axes:
            ax.patches[0].remove()

    def overlay(self, msg, s=18):
        """ Overlay string centered on image. """
        for ax in self.axes:
            ax.images[0].set_alpha(0.5)
            x, y = np.mean(ax.get_xlim()), np.mean(ax.get_ylim())
            ax.text(x, y, msg, color='k', fontsize=s, ha='center', va='center')


class LayerInterface(LayerVisualization):
    """
    Event handler for an individual layer.

    Attributes:
    include (bool) - flag for layer inclusion
    duplicate (bool) - flag for duplicate layer
    exemplar (bool) - flag for exemplary layer
    active_polyhon (bool) - if True, polygon is currently active
    pts (list) - selection boundary points
    traceback (list) - exception traceback
    """

    def __init__(self, layer, axes):
        """
        Instantiate layer interface.

        Args:
        layer (Layer) - image layer
        axes (array like) - axes for blue/red/green color channels
        """

        # call visualization instantiation method
        super().__init__(layer, axes)

        # set layer attributes
        self.include = True
        self.duplicate = False
        self.exemplar = False

        # no initial polygon
        self.active_polygon = False

        # initialize points list
        self.pts = []
        self.traceback = []

    def load(self):
        """ Load layer selection. """
        io = IO()

        # load selected points
        pts = io.read_npy(join(self.path, 'selection.npy'))
        self.pts = pts.tolist()

        # load selection metadata
        md = io.read_json(join(self.path, 'md.json'))
        self.include = md['include']
        self.duplicate = md['duplicate']
        self.exemplar = md['exemplar']

        # add markers
        for pt in self.pts:
             self.add_marker(*pt, color='y', markersize=5)
        self.update_marker('r', markersize=10)

        # add polygon
        if len(self.pts) >= 3:
            self.add_polygon()
            self.active_polygon = True

        # mark if neurons/cones
        if self.include==False and self.duplicate==False:
            self.overlay('NEURONS\n&\nCONES')

        # mark if duplicate
        if self.include==False and self.duplicate==True:
            self.overlay('DUPLICATE')

    def save(self):
        """ Save selected points and selection metadata to file. """
        io = IO()
        pts = np.array(self.pts)
        io.write_npy(join(self.path, 'selection.npy'), pts)
        md = dict(include=self.include,
                  duplicate=self.duplicate,
                  exemplar=self.exemplar)
        io.write_json(join(self.path, 'md.json'), md)

    def clear(self):
        """ Clear all points from layer selection bounds. """
        self.pts = []
        self.clear_markers()
        if len(self.axes[0].patches) > 0:
            self.remove_polygon()
            self.active_polygon = False

    def add_point(self, pt):
        """ Add point to layer selection bounds. """

        # store point
        self.pts.append(pt)

        # update previous marker and add new marker
        self.update_marker(color='y', markersize=5)
        self.add_marker(*pt, color='r', markersize=10)

        # update polygon
        if self.active_polygon:
            self.update_polygon()
        elif len(self.pts) == 3:
            self.add_polygon()
            self.active_polygon = True

    def remove_point(self):
        """ Remove last point added to layer selection bounds. """
        _ = self.pts.pop()
        self.remove_marker()
        self.update_marker(color='r', markersize=10)
        if len(self.pts) < 3:
            self.active_polygon = False

    def update_polygon(self):
        """ Update polygon for each image. """
        self.remove_polygon()
        if self.active_polygon:
            self.add_polygon()

    def undo(self):
        """ Remove last point added and update polygon. """
        self.remove_point()
        self.update_polygon()


class StackInterface:
    """
    Object for visualizing multiple layers in an image stack.

    Attributes:
    path (str) - layer path
    axes (array like) - axes for blue/red/green color channels
    """

    def __init__(self, stack):
        self.path = stack.path
        self.build_interface(stack)

    def load(self):
        """ Load from existing selection data. """
        for interface in layer_to_interface.values():
            interface.load()

    def build_interface(self, stack):
        """
        Build interface by adding interface for each layer.

        Args:
        stack (Stack) - image stack
        """

        # create figure
        self.fig = plt.figure(figsize=(6.75, 2.25*stack.depth))
        gs = GridSpec(nrows=stack.depth, ncols=3, wspace=.01, hspace=.01)

        # instantiate maps
        self.layer_to_interface = {}
        self.ax_to_layer = {}

        # build interface for each layer
        for i, layer in enumerate(stack):

            # create axes for current layer
            ax0 = self.fig.add_subplot(gs[i*3])
            ax1 = self.fig.add_subplot(gs[i*3+1])
            ax2 = self.fig.add_subplot(gs[i*3+2])
            axes=(ax0, ax1, ax2)

            # add layer gui to layer --> interface map
            self.layer_to_interface[i] = LayerInterface(layer, axes)

            # update axis --> layer map
            for ax in axes:
                self.ax_to_layer[ax] = i

            # label top row
            if i == 0:
                ax0.set_title('Disc {}\nDAPI'.format(stack._id), fontsize=14)
                ax1.set_title('Disc {}\nUbiRFP'.format(stack._id), fontsize=14)
                ax2.set_title('Disc {}\nPntGFP'.format(stack._id), fontsize=14)
