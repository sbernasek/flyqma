import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class AlphaVisualization:
    """ Visualization methods for AlphaShapes class. """


    def plot_boundary(self, color='k', ax=None, **kwargs):
        """
        Plot boundary of concave hull as a collection of line objects.

        Args:

            color (str or RGBA) - path color

            ax (matplotlib.axes.AxesSubplot) - if None, create figure

            kwargs: keyword arguments for matplotlib.LineCollection

        """

        # create figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 2))
            xmin, ymin = self.points.min(axis=0)
            xmax, ymax = self.points.max(axis=0)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.axis('off')

        # create line collection and add it to the axis
        lines = LineCollection(self.boundary, colors=color, **kwargs)
        ax.add_collection(lines)

        # format axis
        ax.set_aspect(1)

        return ax


class AlphaShapes(AlphaVisualization):
    """
    Object for computing the concave hull (alpha shape) of a set of points.

    Attributes:

        points (np.ndarray[float]) - points, shape (n,2)

        alpha (float) - alpha value

        only_outer (bool) - if True, only keep outer border

        edge_indices (np.ndarray[int]) - (i,j) pairs representing edges of the alpha-shape. Indices (i,j) index the points array.

    References:

    stackoverflow: questions/50549128/boundary-enclosing-a-given-set-of-points

    """

    def __init__(self, points, alpha, only_outer=True):
        """
        Instantiate alpha shape object.

        Args:

            points (np.ndarray[float]) - points, shape (n,2)

            alpha (float) - alpha value

            only_outer (bool) - if True, only keep outer border

        """

        self.points = points
        self.alpha = alpha
        self.only_outer = only_outer
        edge_indices = self._alpha_shape(points, alpha, only_outer=only_outer)
        self.edge_indices = np.array(list(edge_indices))

    def __call__(self, xy):
        """ Test whether each point in <xy> lies within the alpha shape. """
        f = np.vectorize(lambda x, y: self._is_inside(x, y, self.points, self.edges))
        return f(xy.T)

    @property
    def boundary(self):
        """ Boundary line segments. """
        return self.points[self.edge_indices]

    @staticmethod
    def _alpha_shape(points, alpha, only_outer=True):
        """
        Compute the concave hull (alpha shape) of a set of points.

        Args:

            points (np.ndarray[float]) - points, shape (n,2)

            alpha (float) - alpha value

            only_outer (bool) - if True, only keep outer border

        Returns:

            boundary (list of tuples) - Set of (i,j) pairs representing edges of the alpha-shape. Indices (i,j) index the points array.

        References:

            https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points

        """
        assert points.shape[0] > 3, "Need at least four points"

        def add_edge(edges, i, j):
            """ Add a line between the i-th and j-th points if it's not already in the list. """
            if (i, j) in edges or (j, i) in edges:
                # already added
                assert (j, i) in edges, "Can't go twice over same directed edge right?"
                if only_outer:
                    # if both neighboring triangles are in shape, it's not a boundary edge
                    edges.remove((j, i))
                return
            edges.add((i, j))

        tri = Delaunay(points)
        edges = set()
        # Loop over triangles:
        # ia, ib, ic = indices of corner points of the triangle
        for ia, ib, ic in tri.vertices:
            pa = points[ia]
            pb = points[ib]
            pc = points[ic]
            # Computing radius of triangle circumcircle
            # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
            a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
            c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
            s = (a + b + c) / 2.0
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            circum_r = a * b * c / (4.0 * area)
            if circum_r < alpha:
                add_edge(edges, ia, ib)
                add_edge(edges, ib, ic)
                add_edge(edges, ic, ia)
        return edges

    @staticmethod
    def _is_inside(x, y, points, edges, eps=1.0e-10):
        """
        Check if point (<x>, <y>) lies within the alpha shape defined by <points> and <edges>.
        """
        intersection_counter = 0
        for i, j in edges:
            assert abs((points[i,1]-y)*(points[j,1]-y)) > eps, 'Need to handle these end cases separately'
            y_in_edge_domain = ((points[i,1]-y)*(points[j,1]-y) < 0)
            if y_in_edge_domain:
                upper_ind, lower_ind = (i,j) if (points[i,1]-y) > 0 else (j,i)
                upper_x = points[upper_ind, 0]
                upper_y = points[upper_ind, 1]
                lower_x = points[lower_ind, 0]
                lower_y = points[lower_ind, 1]

                # is_left_turn predicate is evaluated with: sign(cross_product(upper-lower, p-lower))
                cross_prod = (upper_x - lower_x)*(y-lower_y) - (upper_y - lower_y)*(x-lower_x)
                assert abs(cross_prod) > eps, 'Need to handle these end cases separately'
                point_is_left_of_segment = (cross_prod > 0.0)
                if point_is_left_of_segment:
                    intersection_counter = intersection_counter + 1
        return (intersection_counter % 2) != 0
