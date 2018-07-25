import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation
import networkx as nx


class Graph:

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
        pts = df[['centroid_y', 'centroid_x']].values
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

    def apply_distance_filter(self, q=80):
        max_lengths = self._evaluate_max_edge_lengths(self.tri)
        mask = max_lengths > np.percentile(max_lengths, q=q)
        self.tri.set_mask(mask)

    def update_graph(self):
        self.edges = self.node_map(self.tri.edges)
        self.nodes = np.array(sorted(np.unique(self.edges)), dtype=int)

    def plot_edges(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        lines, markers = plt.triplot(self.tri, **kwargs)

    def color_triangles(self, color_by='genotype', agg='mean'):
        """ Assign color to each triangle based on aggregate node values. """

        # get triangulation vertices
        vertices = self.node_map(self.tri.triangles)

        # get value of each node
        get_level = lambda node_id: self.df[color_by].loc[node_id]
        levels = np.apply_along_axis(get_level, axis=1, arr=vertices)

        # aggregate across each triangle
        if agg == 'max':
            colors = levels.max(axis=1)
        elif agg == 'min':
            colors = levels.min(axis=1)
        elif agg == 'std':
            colors = levels.std(axis=1)
        else:
            colors = levels.mean(axis=1)

        return colors

    def plot_triangles(self, ax=None, colors=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        ax.tripcolor(self.tri, facecolors=colors, lw=0, edgecolor='none', antialiased=True, **kwargs)


class WeightedGraph(Graph):

    def __init__(self, df, q=95):
        Graph.__init__(self, df)
        self.apply_distance_filter(q)
        self.q = q
        self.update_graph()
        self.excluded = self.get_excluded_nodes()

    def get_excluded_nodes(self):
        before = self.df.index.values
        after = self.nodes
        return before[~np.isin(before, after)]

    def weight_edges(self, channel='r'):
        wf = WeightFunction(self.df, channel=channel)
        return wf.assess_weights(self.edges)

    def build_links(self, weighted=True, channel='r'):
        """ Returns list of (node_from, node_to, [weight]) tuples. """
        if weighted:
            weights = self.weight_edges(channel)
            unpack = lambda e, w: (int(e[0]), int(e[1]), w)
            links = [unpack(e, w) for e, w in zip(self.edges, weights)]
        else:
            links = [(int(e[0]), int(e[1])) for e in self.edges]
        return links


class WeightFunction:

    def __init__(self, df, channel='r'):
        self.df = df
        self.values = df[channel]

    def difference(self, i, j):
        return np.abs(self.values.loc[i] - self.values.loc[j])

    def scaled_difference(self, i, j):
        x = self.values.loc[i]
        y = self.values.loc[j]
        return np.abs(x-y) / ((x+y)/2)

    def assess_weights(self, edges):
        energy = np.array([self.difference(*e) for e in edges])
        weights = np.exp(-energy/np.mean(energy))
        return weights
