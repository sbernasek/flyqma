from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.tri import Triangulation
import networkx as nx
from collections import Counter

from .triangulation import LocalTriangulation
from .infomap import InfoMap
from .correlation import SpatialCorrelation


class TopologicalProperties:
    """ Topological properties for Graph objects. """

    @property
    def num_nodes(self):
        """ Number of nodes. """
        return len(self.data)

    @property
    def nodes(self):
        """ Unique nodes in graph. """
        return np.array(sorted(np.unique(self.edges)), dtype=int)

    @property
    def nodes_order(self):
        """ Indices that sort nodes by positional index in <self.data>. """
        return np.argsort(self.position_map(self.nodes))

    @property
    def edges(self):
        """ Distance-filtered edges. """
        return self.node_map(self.tri.edges)

    @property
    def edge_list(self):
        """ Distance-filtered edges as (from, to) tuples. """
        return [(int(e[0]), int(e[1]), None) for e in self.edges]

    @property
    def adjacency(self):
        """ Adjacency matrix ordered by <self.nodes>. """
        return nx.to_numpy_array(self.G, nodelist=self.nodes)

    @property
    def adjacency_positional(self):
        """ Adjacency matrix ordered by positional index in <self.data>. """
        return self.adjacency[self.nodes_order, :][:, self.nodes_order]


class SpatialProperties:
    """ Spatial properties for Graph objects. """

    @property
    def node_positions(self):
        """ Assign 2D coordinate positions to nodes. """
        node_positions = {}
        for k in self.nodes:
            i = self.position_map(k)
            node_positions[k] = np.array([self.tri.x[i], self.tri.y[i]])
        return node_positions

    @property
    def node_positions_arr(self):
        """ N x 2 array of node coordinates, ordered by positional index."""
        return self.data[self.xykey].values

    @property
    def edge_lengths(self):
        """ Unique edge lengths. """
        return sorted(self.distance_matrix[self.adjacency.nonzero()])[::2]

    @property
    def median_edge_length(self):
        """ Median edge length. """
        return np.median(self.edge_lengths)

    @staticmethod
    def _distance_matrix(xy):
        """ Returns euclidean distance matrix between all points in <xy>. """

        x, y = xy.T
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        x_component = np.repeat(x**2, x.size, axis=1) + np.repeat(x.T**2, x.size, axis=0) - 2*np.dot(x, x.T)
        y_component = np.repeat(y**2, y.size, axis=1) + np.repeat(y.T**2, y.size, axis=0) - 2*np.dot(y, y.T)

        return np.sqrt(x_component + y_component)

    @property
    def distance_matrix(self):
        """ Euclidean distance matrix between all nodes. """
        return self._distance_matrix(self.node_positions_arr)

    @staticmethod
    def get_matrix_upper(matrix):
        """
        Return upper triangular portion of a 2-D matrix.

        Parameters:

            matrix (2D np.ndarray)

        Returns:

            upper (1D np.ndarray) - upper triangle, ordered row then column

        """
        return matrix[np.triu_indices(len(matrix), k=1)]

    @property
    def unique_distances(self):
        """ Upper triangular portion of euclidean distance matrix. """
        return self.get_matrix_upper(self.distance_matrix)

    @staticmethod
    def evaluate_fluctuations(values):
        """
        Construct pairwise fluctuation matrix for <values>.

        Args:

            values (1D np.ndarray[float]) - attribute values

        Returns:

            fluctuations (2D np.ndarray[float]) - pairwise fluctuations

        """

        # compute correlation matrix and return upper triangular portion
        v = (values - values.mean())
        fluctuations = np.dot(v, v.T) / np.var(values)
        return fluctuations

    def get_fluctuations_matrix(self, attribute, log=True):
        """
        Returns normalized pairwise fluctuations of <attribute> value for each node in the graph.

        Args:

            attribute (str) - name of attribute

            log (bool) - if True, log-transform attribute values

        Returns:

            fluctuations (2D np.ndarray[float]) - pairwise fluctuations

        """
        attribute_values = self.data[attribute].values.reshape(-1, 1)
        if log:
            attribute_values = np.log(attribute_values)
        return self.evaluate_fluctuations(attribute_values)

    def _mark_boundary(self, attribute='genotype', max_edges=0):
        """
        Mark boundaries between nodes with differing attribute values. Boundaries are identified by nodes that share an edge with another node whose attribute value differs.

        Args:

            attribute (str) - used to identify distinct groups of nodes

            max_edges (int) - max number of transgroup edges for interior nodes

        Returns:

            interior (np.vector[bool]) - mask for interior points

        """

        # assign genotype to edges
        assign_genotype = np.vectorize(dict(self.data[attribute]).get)
        edge_genotypes = assign_genotype(self.edges)

        # find edges traversing clones
        boundaries = (edge_genotypes[:, 0] != edge_genotypes[:, 1])

        # get number of clone-traversing edges per node
        boundary_edges = self.edges[boundaries]
        edge_counts = Counter(boundary_edges.ravel())

        # assign boundary label to nodes with too many clone-traversing edges
        boundary_nodes = [n for n, c in edge_counts.items() if c>max_edges]
        interior = np.ones(self.num_nodes, dtype=bool)
        interior[self.position_map(boundary_nodes)] = False

        return interior


class GraphVisualizationMethods:
    """ Methods for visualizing a Graph instance. """

    def plot_edges(self, ax=None, **kwargs):
        """
        Plot triangulation edges.

        Args:

            ax (matplotlib.axes.AxesSubplot)

            kwargs: keyword arguments for matplotlib.pyplot.triplot

        """
        if ax is None:
            fig, ax = plt.subplots()
        lines, markers = plt.triplot(self.tri, **kwargs)

    def label_triangles(self, label_by='genotype'):
        """
        Label each triangle with most common node attribute value.

        Args:

            label_by (str) - node attribute used to label each triangle

        Returns:

            labels (np.ndarray[int]) - labels for each triangle

        """

        # get triangulation vertices
        vertices = self.node_map(self.tri.triangles)

        # get value of each node
        get_level = lambda node_id: self.data[label_by].loc[node_id]
        levels = np.apply_along_axis(get_level, axis=1, arr=vertices)

        # aggregate within triangles
        get_mode = lambda x: Counter(x).most_common(1)[0][0]
        labels = np.apply_along_axis(get_mode, axis=1, arr=levels)

        return labels

    def plot_triangles(self,
                       label_by='genotype',
                       cmap=None,
                       ax=None,
                       **kwargs):
        """
        Plot triangle faces using tripcolor.

        Args:

            label_by (str) - data attribute used to color each triangle

            cmap (matplotlib.colors.ColorMap) - colormap for attribute values

            ax (matplotlib.axes.AxesSubplot)

            kwargs: keyword arguments for plt.tripcolor

        """
        if ax is None:
            fig, ax = plt.subplots()

        # assign triangle colors
        colors = self.label_triangles(label_by=label_by)

        # define colormap
        if cmap is None:
            cmap = ListedColormap(['y', 'm', 'k'], N=3)

        # plot triangle faces
        ax.tripcolor(self.tri,
                     facecolors=colors,
                     cmap=cmap,
                     lw=0,
                     edgecolor='none',
                     antialiased=True,
                     **kwargs)

    def show(self, ax=None, colorby=None, disconnect=False, **kwargs):
        """
        Visualize graph.

        Args:

            ax (matplotlib.axes.AxesSubplot) - if None, create figure

            colorby (str) - node attribute used to assign node/edge colors

            disconnect (bool) - if True, remove edges between nodes whose colorby values differ

            kwargs: keyword arguments for NetworkxGraphVisualization.draw

        """
        if colorby is not None:
            msg = 'Colorby attribute must be an integer type.'
            assert self.data[colorby].dtype in (np.integer, int), msg

        # construct graph
        if colorby is not None:
            G = self.get_networkx(colorby)
        else:
            G = self.G

        # disconnect nodes not sharing the <colorby> attribute
        if disconnect:
            is_different = lambda u,v: G.node[u][colorby] != G.node[v][colorby]
            removed_edges = [edge for edge in G.edges if is_different(*edge)]
            G.remove_edges_from(removed_edges)

        # draw graph
        vis = NetworkxGraphVisualization(G, self.node_positions)
        vis.draw(ax=ax, colorby=colorby, **kwargs)


class CommunityDetection:
    """ Methods for detecting communities in a Graph. """

    def detect_communities(self, **kwargs):
        """
        Detect communities using InfoMap clustering.

        Accepts keyword arguments for InfoMap, including:

            twolevel (bool) - if True, perform two-level clustering, otherwise defaults to multi-level clustering

            N (int) - number of trials

        """
        self.imap = InfoMap(self.edge_list, **kwargs)

    def _assign_community(self, level=None):
        """
        Assign communities using InfoMap clustering.

        Args:

            level (int) - module level at which aggregation occurs, starting from the finest resolution

        Returns:

            labels (np.ndarray[int]) - community labels ordered by positional index

        """
        return self.imap(self.nodes[self.nodes_order], level)

    def assign_community(self, level=None, key='community'):
        """
        Assign communities using InfoMap clustering.

        Args:

            level (int) - module level at which aggregation occurs, starting from the finest resolution

            key (str) - name of community attribute

        """
        self.data[key] = self._assign_community(level=level)


class Graph(TopologicalProperties,
            SpatialProperties,
            CommunityDetection,
            GraphVisualizationMethods):
    """
    Object provides an undirected unweighted graph connecting adjacent cells.

    Attributes:

        data (pd.DataFrame) - cell measurement data (nodes)

        xykey (list) - attribute keys for node x/y positions

        G (nx.Graph) - undirected graph instance

        nodes (np.ndarray[int]) - node indices

        edges (np.ndarray[int]) - pairs of connected node indices

        node_map (vectorized func) - maps positional index to node index

        position_map (vectorized func) - maps node index to positional index

        tri (matplotlib.tri.Triangulation) - triangulation of node positions

    """

    def __init__(self, data, xykey=None):
        """
        Instantiate Graph object.

        Args:

            data (pd.DataFrame) - cell measurement data (nodes)

            xykey (list) - attribute keys for node x/y positions

        """

        # set xykey
        if xykey is None:
            xykey = ['centroid_x', 'centroid_y']
        self.xykey = xykey

        # store data
        self.data = data

        # instantiate community detection
        self.community_labels = None
        self.imap = None

        # define mapping from position to node index
        position_to_node = dict(enumerate(data.index))
        self.node_map = np.vectorize(position_to_node.get)

        # define reverse map
        node_to_position = {v: k for k, v in position_to_node.items()}
        self.position_map = np.vectorize(node_to_position.get)

        # triangulate
        self.tri = self._construct_triangulation(data, xykey)

        # build networkx graph instance
        self.G = self.get_networkx()

    def copy(self):
        """ Returns deep copy of graph instance. """
        return deepcopy(self)

    def get_subgraph(self, ind):
        """ Instantiate subgraph from DataFrame indices. """
        return Graph(self.data.loc[ind], xykey=self.xykey)

    def get_networkx(self, *node_attributes):
        """
        Returns networkx instance of graph.

        Args:

            node_attributes (str) - attributes to be added for each node

        """
        G = nx.Graph()
        G.add_weighted_edges_from(self.edge_list)

        # add node attributes
        for attr in node_attributes:
            if attr is not None:
                values_dict = dict(self.data.loc[self.nodes][attr])
                nx.set_node_attributes(G, name=attr, values=values_dict)

        return G

    def get_correlations(self, attribute, log=True):
        """
        Returns SpatialCorrelation object for <attribute>.

        Args:

            attribute (str) - name of attribute

            log (bool) - if True, log-transform attribute values

        Returns:

            correlations (SpatialCorrelation)

        """
        d_ij = self.unique_distances
        C = self.get_fluctuations_matrix(attribute, log=log)
        C_ij = self.get_matrix_upper(C)
        return SpatialCorrelation(d_ij, C_ij)

    @staticmethod
    def _construct_triangulation(data, xykey, **kwargs):
        """
        Construct Delaunay triangulation with edge filter.

        Args:

            data (pd.DataFrame) - node measurement data

            xykey (list) - attribute keys for node x/y coordinates

            kwargs: keyword arguments for triangulation

        """
        pts = data[xykey].values
        return LocalTriangulation(*pts.T, **kwargs)


class WeightedGraph(Graph):
    """
    Object provides an undirected weighted graph connecting adjacent cells. Edge weights are evaluated based on the similarity of expression between pairs of connected nodes. Node similariy is based on the cell measurement data attribute specified by the 'weighted_by' parameter.

    Attributes:

        weighted_by (str) - data attribute used to weight edges

        imap (spatial.InfoMap) - community detection

        community_labels (np.ndarray[int]) - community label for each node

        logratio (bool) - if True, weight edges by log ratio

        distance (bool) - if True, weights edges by distance rather than similarity

    Inherited attributes:

        data (pd.DataFrame) - cell measurement data (nodes)

        xykey (list) - attribute keys for node x/y positions

        nodes (np.ndarray[int]) - node indices

        edges (np.ndarray[int]) - pairs of connected node indices

        node_map (vectorized func) - maps positional index to node index

        position_map (vectorized func) - maps node index to positional index

        tri (matplotlib.tri.Triangulation) - triangulation of node positions

    """

    def __init__(self, data, weighted_by,
                 xykey=None,
                 logratio=True,
                 distance=False):
        """
        Instantiate weighted graph.

        Args:

            data (pd.DataFrame) - cell measurement data

            weighted_by (str) - data attribute used to weight edges

            xykey (list) - attribute keys for node x/y positions

            logratio (bool) - if True, weight edges by log ratio

            distance (bool) - if True, weights edges by distance

        """

        # set attributes
        self.weighted_by = weighted_by
        self.logratio = logratio
        self.distance = distance

        # instantiate graph
        super().__init__(data, xykey)

    @property
    def edge_list(self):
        """ Distance-filtered edges as (from, to, weight) tuples. """
        if self.weighted_by not in (None, 'none', 'None'):
            weights = self.evaluate_edge_weights()
            unpack = lambda e, w: (int(e[0]), int(e[1]), w)
            edge_list = [unpack(e, w) for e, w in zip(self.edges, weights)]
        else:
            edge_list = super().edge_list

        return edge_list

    def evaluate_edge_weights(self):
        """
        Evaluate edge weights.

        Returns:

            weights (np.ndarray[float]) - edge weights

        """
        wf = WeightFunction(self.data,
                            weighted_by=self.weighted_by,
                            distance=self.distance)
        return wf.assess_weights(self.edges, logratio=self.logratio)


class WeightFunction:
    """
    Object for weighting graph edges by similarity.

    Attributes:

        data (pd.DataFrame) - nodes data

        weighted_by (str) - node attribute used to assess similarity

        values (pd.Series) - node attribute values

        distance (bool) - if True, weights edges by distance

    """

    def __init__(self, data, weighted_by='r', distance=False):
        """
        Instantiate edge weighting function.

        Args:

            data (pd.DataFrame) - nodes data

            weighted_by (str) - node attribute used to assess similarity

            distance (bool) - if True, weights edges by distance

        """
        self.data = data
        self.weighted_by = weighted_by
        self.values = data[weighted_by]
        self.distance = distance

    def difference(self, i, j):
        """
        Evaluate difference in values between nodes i and j.

        Args:

            i, j (ind) - node indices

        Returns:

            difference (float)

        """
        return np.abs(self.values.loc[i] - self.values.loc[j])

    def logratio(self, i, j):
        """
        Evaluate log ratio between nodes i and j.

        Args:

            i, j (ind) - node indices

        Returns:

            logratio (float)

        """
        return np.abs(np.log(self.values.loc[i]/self.values.loc[j]))

    def assess_weights(self, edges, logratio=False):
        """
        Evaluate edge weights normalized by mean difference in node values.

        Args:

            edges (list of (i, j) tuples) - edges between nodes i and j

            logratio (bool) - if True, weight edges by logratio

        Returns:

            weights (np.ndarray[float]) - edge weights

        """
        if logratio:
            energy = np.array([self.logratio(*e) for e in edges])
        else:
            energy = np.array([self.difference(*e) for e in edges])
        weights = np.exp(-energy/np.mean(energy))

        # invert similarities to distances
        if self.distance:
            weights = 1 - weights

        return weights


class NetworkxGraphVisualization:
    """
    Object for visualizing a NetworkX Graph object.

    Attributes:

        G (nx.Graph) - networkx graph object

        pos (np.ndarray[float]) - 2D node positions

    """

    def __init__(self, G, pos):
        """
        Instantiate graph visualization.

        Args:

            G (nx.Graph) - networkx graph object

            pos (np.ndarray[float]) - 2D node positions

        """
        self.G = G
        self.pos = pos

    def build_cmap(self, colorby):
        """ Build colormap. """
        levels = [v for k,v in nx.get_node_attributes(self.G, colorby).items()]
        N = np.unique(levels).size
        colors = np.random.random(size=(N, 3))
        return ListedColormap(colors, 'indexed', N)

    def draw(self,
             ax=None,
             colorby=None,
             edge_color='k',
             node_color='k',
             cmap=None,
             **kwargs):
        """
        Draw graph.

        Args:

            ax (matplotlib.axes.AxesSubplot) - axis on which to draw graph

            colorby (str) - node attribute on which nodes/edges are colored

            edge_color, node_color (str) - edge/node colors, overrides colorby

            node_cmap (matplotlib.colors.ColorMap) - node colormap

        """

        assert (colorby is not None or node_color is not None), 'Either node color or colorby attribute must be specified.'
        assert (colorby is not None or edge_color is not None), 'Either edge color or colorby attribute must be specified.'

        # create figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # build colormap
        if cmap is None and None in (edge_color, node_color):
            cmap = self.build_cmap(colorby=colorby)

        # assign node color
        if node_color is None:
            get_color = lambda x: cmap(int(x))
            node_colors = [get_color(v) for k, v in nx.get_node_attributes(self.G, colorby).items()]
        else:
            node_colors = [node_color for _ in self.G.nodes]

        # assign edge color
        if edge_color is None:
            node_value = lambda index: self.G.node[index][colorby]
            node_color = lambda index: cmap(node_value(index))
            edge_colors = [node_color(u) for u,v in self.G.edges]
        else:
            edge_colors = [edge_color for _ in self.G.edges]

        # draw graph
        self._draw(ax, self.G, self.pos, node_colors, edge_colors, **kwargs)

    @staticmethod
    def _draw(ax,
              G,
              pos,
              node_colors,
              edge_colors,
              node_alpha=1,
              node_size=20,
              node_edgewidth=0.,
              lw=3,
              edge_alpha=0.5,
              **kwargs):
        """
        Draw graph.

        Args:

            G (nx.Graph) - graph object

            pos (np.ndarray[float]) - node xy positions in space (graph layout)

            node_colors (array like) - node values

            edge_colors (array like) - edge colors

            node_alpha (float) - node transparency

            node_size (float) - node size

            node_edgewidth (float) - linewidth of node outline

            lw (float) - maximum edge linewidth

            edge_alpha (float) - edge line transparency

            kwargs: keyword arguments for nx.draw_networkx_nodes

        """

        # extract edge weights
        edge_weights = [x[-1]['weight'] for x in G.edges(data=True)]

        # draw edges
        norm = Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
        edge_widths = [norm(w)*lw for w in edge_weights]
        nx.draw_networkx_edges(G, pos,
                               ax=ax,
                               edge_color=edge_colors,
                               width=edge_widths,
                               alpha=edge_alpha)

        # draw nodes
        nodeCollection = nx.draw_networkx_nodes(G,
            pos=pos,
            ax=ax,
            node_color=node_colors,
            node_alpha=node_alpha,
            node_size=node_size,
            linewidths=node_edgewidth,
            **kwargs)

        ax.axis('off')
