import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.tri import Triangulation
import networkx as nx
from collections import Counter

from .infomap import InfoMap


class Graph:
    """
    Object representing an undirected weighted graph connecting adjacent cells.

    Attributes:
    df (pd.DataFrame) - cell measurement data (nodes)
    nodes (np.ndarray[int]) - node indices
    edges (np.ndarray[int]) - pairs of connected node indices
    node_map (vectorized func) - maps positional index to node index
    position_map (vectorized func) - maps node index to positional index
    tri (matplotlib.tri.Triangulation) - triangulation of node positions
    """

    def __init__(self, df):
        self.df = df
        self.nodes = df.index.values

        # define mapping from position to node index
        position_to_node = dict(enumerate(df.index))
        self.node_map = np.vectorize(position_to_node.get)

        # define reverse map
        node_to_position = {v: k for k, v in position_to_node.items()}
        self.position_map = np.vectorize(node_to_position.get)

        # triangulate
        self.tri = self._construct_triangulation(df)
        self.update_graph()

    def get_networkx(self):
        G = nx.Graph()
        G.add_edges_from(self.edges)
        return G

    def get_subgraph(self, ind):
        """ Instantiate subgraph from DataFrame indices. """
        return Graph(self.df.loc[ind])

    @staticmethod
    def _construct_triangulation(df):
        """ Construct Delaunay triangulation. """
        pts = df[['centroid_x', 'centroid_y']].values
        return Triangulation(*pts.T)

    @staticmethod
    def _evaluate_max_edge_lengths(tri):
        distances = []
        x, y = tri.x, tri.y
        for points in tri.triangles:
            a,b,c = points
            d0 = np.sqrt( (x[a] - x[b]) **2 + (y[a] - y[b])**2 )
            d1 = np.sqrt( (x[b] - x[c]) **2 + (y[b] - y[c])**2 )
            d2 = np.sqrt( (x[c] - x[a]) **2 + (y[c] - y[a])**2 )
            max_edge = max([d0, d1, d2])
            distances.append(max_edge)
        return np.array(distances)

    def apply_distance_filter(self, q=95):
        """ Mask edges with lengths exceeding specified quantile. """
        max_lengths = self._evaluate_max_edge_lengths(self.tri)
        mask = max_lengths > np.percentile(max_lengths, q=q)
        self.tri.set_mask(mask)

    def update_graph(self):
        """ Update node list based on included edges. """
        self.edges = self.node_map(self.tri.edges)
        self.nodes = np.array(sorted(np.unique(self.edges)), dtype=int)

    def plot_edges(self, ax=None, **kwargs):
        """ Plot triangulation. """
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
        get_level = lambda node_id: self.df[label_by].loc[node_id]
        levels = np.apply_along_axis(get_level, axis=1, arr=vertices)

        # aggregate within triangles
        get_mode = lambda x: Counter(x).most_common(1)[0][0]
        labels = np.apply_along_axis(get_mode, axis=1, arr=levels)

        return labels

    def plot_triangles(self, ax=None, colors=None, **kwargs):
        """ Plot triangle faces using tripcolor. """
        if ax is None:
            fig, ax = plt.subplots()
        ax.tripcolor(self.tri, facecolors=colors, lw=0, edgecolor='none', antialiased=True, **kwargs)

    def show(self, ax, **kw):
        """ Visualize graph. """
        vis = GraphVisualization.from_graph(self)
        vis.draw(ax=ax, **kw)


class WeightedGraph(Graph):

    def __init__(self, df, weighted_by='r_normalized', q=95):
        Graph.__init__(self, df)
        self.apply_distance_filter(q)
        self.q = q
        self.update_graph()
        self.excluded = self.get_excluded_nodes()
        self.weighted_by = weighted_by
        self.community_labels = None

    def get_excluded_nodes(self):
        before = self.df.index.values
        after = self.nodes
        return before[~np.isin(before, after)]

    def weight_edges(self, weighted_by='r'):
        wf = WeightFunction(self.df, weighted_by=weighted_by)
        return wf.assess_weights(self.edges)

    def build_links(self):
        """ Returns list of (node_from, node_to, [weight]) tuples. """
        if self.weighted_by not in (None, 'none', 'None'):
            weights = self.weight_edges(self.weighted_by)
            unpack = lambda e, w: (int(e[0]), int(e[1]), w)
            links = [unpack(e, w) for e, w in zip(self.edges, weights)]
        else:
            links = [(int(e[0]), int(e[1])) for e in self.edges]
        return links

    def find_communities(self, **kw):
        """ Find communities. """
        edges = self.build_links()
        community_detector = InfoMap(edges, **kw)
        self.community_labels = community_detector(self.nodes)


class WeightFunction:
    """
    Object for weighting graph edges by similarity.

    Attributes:
    df (pd.DataFrame) - nodes data
    weighted_by (str) - node attribute used to assess similarity
    values (pd.Series) - node attribute values
    """

    def __init__(self, df, weighted_by='r'):
        """
        Instantiate edge weighting function.

        Args:
        df (pd.DataFrame) - nodes data
        weighted_by (str) - node attribute used to assess similarity
        """
        self.df = df
        self.weighted_by = weighted_by
        self.values = df[weighted_by]

    def difference(self, i, j):
        """
        Evaluate difference in values between nodes i and j.

        Args:
        i, j (ind) - node indices

        Returns:
        difference (float)
        """
        return np.abs(self.values.loc[i] - self.values.loc[j])

    def assess_weights(self, edges):
        """
        Evaluate edge weights normalized by mean difference in node values.

        Args:
        edges (list of (i, j) tuples) - edges between nodes i and j

        Returns:
        weights (np.ndarray[float]) - edge weights
        """
        energy = np.array([self.difference(*e) for e in edges])
        weights = np.exp(-energy/np.mean(energy))
        return weights


class GraphVisualization:
    """
    Object for graph visualization.

    Attributes:
    G (nx.Graph) - networkx graph object
    pos (np.ndarray[float]) - 2D node positions
    """

    def __init__(self, G, pos):
        self.G = G
        self.pos = pos

    @classmethod
    def from_graph(cls, graph):
        """
        Instantiate from Graph instance.

        Args:
        graph (Graph)

        Returns:
        nxGraph (GraphVisualization)
        """

        # build graph from edges
        edges = [cls.to_nx_edge(*edge) for edge in graph.build_links()]
        G = cls.build_graph(edges)

        # assign node positions
        pos = cls.assign_node_positions(graph)

        # assign two-level InfoMap communities
        nx.set_node_attributes(G, name='community', values=InfoMap(graph).node_to_module)

        try:
            gs = graph.df.loc[graph.nodes].genotype
            node_to_genotype = {k: v for k, v in zip(gs.index, gs)}
            nx.set_node_attributes(G, name='genotype', values=node_to_genotype)
        except:
            raise ValueError('No genotype attribute found.')

        try:
            node_to_km_label = dict(graph.df.km_label)
            nx.set_node_attributes(G, name='km_label', values=node_to_km_label)
        except:
            raise ValueError('No km_label attribute found.')

        return GraphVisualization(G, pos)

    @staticmethod
    def build_graph(edges):
        """ Build networkx graph object. """
        G = nx.Graph()
        G.add_edges_from(edges)
        return G

    @staticmethod
    def to_nx_edge(x, y, weight):
        """ Convert edge to networkx format.  """
        return (x, y, dict(weight=weight))

    @staticmethod
    def assign_node_positions(graph):
        """ Assign 2D coordinate positions to nodes. """
        node_positions = {}
        for k in graph.nodes:
            i = graph.position_map(k)
            node_positions[k] = np.array([graph.tri.x[i], graph.tri.y[i]])
        return node_positions

    def set_cmap(self, colorby='community'):
        """ Set colormap. """
        levels = [v for k,v in nx.get_node_attributes(self.G, colorby).items()]
        N = max(levels) + 1
        colors = np.random.random(size=(N, 3))
        return ListedColormap(colors, 'indexed', N)

    def get_edge_weights(self, label_on, disconnect=True):
        """
        Get edge weights.

        Args:
        label_on (str) - attribute used to determine node kinship
        disconnect (bool) - if True, exclude edges between communities

        Returns:
        weights (list) - edge weights
        """
        weights = []
        for u, v in self.G.edges:
            if self.G.node[u][label_on]==self.G.node[v][label_on]:
                weights.append(self.G[u][v]['weight'])
            elif not disconnect:
                weights.append(self.G[u][v]['weight'])
            else:
                continue
        return weights

    def draw(self,
             ax=None,
             colorby='community',
             disconnect=False,
             ec='k',
             node_cmap=None,
             **kw):
        """
        Draw graph.

        Args:
        ax (matplotlib.axes.AxesSubplot) - axis on which to draw graph
        colorby (str) - node attribute on which nodes are colored
        disconnect (bool) - if True, exclude edges between communities
        ec (str) - edge color
        node_cmap (matplotlib.colors.ColorMap) - node colormap
        """

        # create figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 15))

        # set colormap
        if node_cmap is None:
            node_cmap = self.set_cmap(colorby=colorby)
        self.cmap = node_cmap

        # get edge properties
        if colorby == 'genotype':
            edge_colors = [self.cmap(self.G.node[u][colorby]) for u,v in self.G.edges]
        else:
            edge_colors = [ec for _ in self.G.edges]
        edge_weights = self.get_edge_weights(colorby, disconnect)

        # get node properties
        node_colors = [int(v) for k, v in nx.get_node_attributes(self.G, colorby).items()]

        # draw graph
        self._draw(ax, self.G, self.pos, node_colors, edge_colors, edge_weights, cmap=self.cmap, **kw)

    @staticmethod
    def _draw(ax,
              G,
              pos,
              node_colors,
              edge_colors,
              edge_weights,
              cmap=None,
              node_alpha=1,
              node_size=20,
              lw=3,
              edge_alpha=0.5,
              **kw):
        """
        Draw graph.
        """

        # draw edges
        norm = Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
        edge_widths = [norm(w)*lw for w in edge_weights]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=edge_alpha)

        # draw nodes
        nodeCollection = nx.draw_networkx_nodes(G,
            pos=pos,
            node_color=node_colors,
            node_alpha=node_alpha,
            node_size=node_size,
            cmap=cmap, **kw)

        ax.axis('off')
