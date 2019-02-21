import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial import Voronoi
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

from .spatial.alpha import AlphaShapes


class CloneBoundaries:
    """
    Object for drawing paths around clone boundaries.

    Attributes:

        shapes (list of AlphaShapes) - shape indices for ordered labels

    """

    def __init__(self, graph, label_by='genotype', alpha=50):
        """
        Instantiate clone boundary object.

        Args:

            graph (spatial.Graph)

            label_by (str) - attribute used to label clones

            alpha (float) - shape parameter for alpha shapes

        """
        self.shapes = self.build_shapes(graph, label_by, alpha)

    @classmethod
    def from_layer(cls, layer, label_by='genotype', **kwargs):
        """ Instantiate from clones.Layer instance. """
        return cls(layer.graph, label_by=label_by, **kwargs)

    @property
    def norm(self):
        """ Normalization for number of shapes. """
        return Normalize(vmin=0, vmax=len(self.shapes)-1)

    @staticmethod
    def build_shapes(graph, label_by, alpha):
        """ Compile shapes. """
        shapes_dict = {}
        for label in graph.data[label_by].unique():
            xy = graph.data[graph.data[label_by]==label][graph.xykey].values
            shapes_dict[label] = AlphaShapes(xy, alpha=alpha)
        return shapes_dict

    def plot_boundary(self, label, **kwargs):
        """ Plot boundary for clones with <label>. """
        shape = self.shapes[label]
        shape.plot_boundary(**kwargs)

    def plot_boundaries(self, cmap=plt.cm.viridis, **kwargs):
        """ Plot all clone boundaries. """
        for label in self.shapes.keys():
            color = cmap(self.norm(label))
            self.plot_boundary(label, color=color, **kwargs)


class Tessellation:
    """
    Object for visualizing Voronoi tessellations.
    """

    def __init__(self, xy, labels, q=90, colors=None):

        self.vor = Voronoi(xy)
        self.vor.regions = np.array(self.vor.regions)
        self.set_region_mask(q=q)
        self.region_labels = self.label_regions(labels)
        self.verts = self.vor.regions[self.mask]
        self.set_cmap(colors)

    def label_regions(self, labels):
        points = np.argsort(self.vor.point_region)
        point_to_label = np.vectorize(dict(enumerate(labels)).get)
        region_labels = point_to_label(points)
        return region_labels

    def set_cmap(self, colors=None):
        N = len(set(self.region_labels))
        if colors is None:
            colors = np.random.random((N, 3))
        self.cmap = ListedColormap(colors, 'indexed', N)

    @staticmethod
    def _evaluate_area(x, y):
        """ Evaluate area enclosed by a set of points. """
        return 0.5*np.abs(np.dot(x, np.roll(y,1))-np.dot(y, np.roll(x,1)))

    def evaluate_region_area(self, region):
        """ Evaluate pixel area enclosed by a region. """
        return self._evaluate_area(*self.vor.vertices[region, :].T)

    def set_region_mask(self, q=90):
        """
        Mask regions with pixel areas larger than a specified quantile.

        Args:
        q (float) - maximum region area quantile, 0 to 100
        """
        f = np.vectorize(lambda x: -1 not in x and len(x) > 0)
        mask = f(self.vor.regions)
        mask *= self.build_region_area_mask(q=q)
        self.mask = mask

    def build_region_area_mask(self, q=90):
        """
        Mask regions with pixel areas larger than a specified quantile.

        Args:

            q (float) - maximum region area quantile, 0 to 100

        Returns:

            mask (np.ndarray[bool]) - True for regions smaller than maximum area

        """
        evaluate_area = np.vectorize(lambda x: self.evaluate_region_area(x))
        areas = evaluate_area(self.vor.regions)
        threshold = np.percentile(areas, q=q)
        return (areas <= threshold)

    @staticmethod
    def _show(vertices, c='k', ax=None, alpha=0.5):
        """ Visualize vertices. """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlim(0, 2048)
            ax.set_ylim(0, 2048)
            ax.axis('off')
        poly = PolyCollection(vertices)
        poly.set_facecolors(c)
        poly.set_alpha(alpha)
        ax.add_collection(poly)

    def show(self, ax=None, **kw):
        """ Visualize vertices. """
        get_vertices = np.vectorize(lambda region: self.vor.vertices[region])
        vertices = [self.vor.vertices[r] for r in self.vor.regions[self.mask]]
        c = self.cmap(self.region_labels[self.mask[1:]])
        self._show(vertices, c=c, ax=ax, **kw)


class CloneVisualization(Tessellation):
    """
    Object for visualizing clones by shading Voronoi cells.
    """

    def __init__(self, graph, label='genotype', **kw):
        labels = graph.data[label].values
        Tessellation.__init__(self, graph.node_positions_arr, labels, **kw)
