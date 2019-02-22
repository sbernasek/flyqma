import numpy as np
import networkx as nx
from copy import deepcopy

from ...visualization import *

from .infomap import InfoMap


class NeighborSampler:
    """
    Class for sampling node attributes averaged over neighbors.

    Attributes:

        graph (spatial.Graph) - graph instance

        G (nx.Graph) - graph with node attribute

        attr (str) - attribute to be averaged over neighbors

        depth (int) - maximum number of edges connecting neighbors

        log (bool) - if True, log-transform values before averaging

    """

    def __init__(self, graph, attr, depth=1, log=True):
        """
        Instantiate sampler for averaging <attr> value over all neighbors within <depth> of each node.

        Args:

            graph (spatial.Graph) - graph instance

            attr (str) - attribute to be averaged over neighbors

            depth (int) - maximum number of edges connecting neighbors

            log (bool) - if True, log-transform values before averaging

        """
        self.graph = graph
        self.attr = attr
        self.depth = depth
        self.log = log

        # add node attribute to graph
        if self.attr_used not in list(self.G.nodes(data=True)[0].keys()):
            self.add_attribute_to_graph()

        # sample over neigbors
        self.average_over_neighbors()

    @classmethod
    def multisample(cls, attr, *graphs, **kwargs):
        """
        Generate composite sample from one or more <graphs>.

        Args:

            attr (str) - attribute to be averaged over neighbors

            graphs (spatial.Graph) - one or more graph instances

            kwargs: keyword arguments for sampler

        Returns:

            sample (np.ndarray[float]) - 2D array of sampled values, first column contains cell measurements while the second column contains measurements averaged over the neighbors of each cell

            keys (list of str) - attribute keys for sampled data

        """
        samples = []
        for graph in graphs:
            sampler = cls(graph, attr, **kwargs)
            samples.append(sampler.sample)
        return np.vstack(samples), sampler.keys

    @property
    def num_nodes(self):
        """ Number of nodes. """
        return self.graph.num_nodes

    @property
    def data(self):
        """ Graph data. """
        return self.graph.data

    @property
    def G(self):
        """ NetworkX graph instance. """
        return self.graph.G

    @property
    def node_values(self):
        """ Vector of attribute values for each node. """
        values = self.data[self.attr].values
        if self.log:
            values = np.log(values)
        return values

    @property
    def node_values_dict(self):
        """ Dictionary of attribute values, keyed by node index. """
        values = self.data[self.attr]
        if self.log:
            values = np.log(values)
        return dict(values)

    @property
    def neighbors(self):
        """ Dictionary of neighbor indices keyed by node indices. """
        kwargs = dict(depth_limit=self.depth)
        bfs = lambda n: [e[1] for e in nx.bfs_edges(self.G, n, **kwargs)]
        return {node: bfs(node) for node in self.G.nodes}

    @property
    def size_attr(self):
        """ Neighborhood size attribute name. """
        return 'd{:d}_neighbors'.format(self.depth)

    @property
    def attr_used(self):
        """ Name of attribute used to access graph data. """
        if self.log:
            return 'log_' + self.attr
        else:
            return self.attr

    @property
    def averaged_attr(self):
        """ Name of averaged attribute. """
        return '{:s}_d{:d}'.format(self.attr, self.depth)

    @property
    def keys(self):
        """ List of attribute names. """
        return [self.attr, self.averaged_attr]

    @property
    def sample(self):
        """ Returns bivariate sample combining each node's attribute value with the average attribute value in its neighborhood. """
        return self.data[self.keys].values

    def add_attribute_to_graph(self):
        """ Add attribute to networkx graph object. """
        nx.set_node_attributes(self.G, self.node_values_dict, name=self.attr_used)

    @staticmethod
    def _neighbor_average(G, node_values, depth=1):
        """
        Average attribute value over all neighbors of each node in a graph.

        Args:

            G (nx.Graph) - graph with nodes containing an attribute

            node_values (dict) - node attribute values keyed by node index

            depth (int) - maximum number of edges connecting neighbors

        Returns:

            node_to_neighborhood (dict) - dictionary of neighborhood (<attribute>, size) tuples keyed by node index

        """

        # define breadth first search of each <src> node
        bfs = lambda n: [e[1] for e in nx.bfs_edges(G, n, depth_limit=depth)]

        # define function to parse mean node attribute within neighborhood
        parse = lambda nbs: (np.mean([node_values[n] for n in nbs]), len(nbs))

        return {src: parse(bfs(src)) for src in G.nodes}

    def average_over_neighbors(self):
        """
        Average attribute value over all neighbors adjacent to each node.
        """

        # average attribute over neighbors
        neighbor_dict = self._neighbor_average(self.G, self.node_values_dict, self.depth)

        # extract average and sample size for each node
        keys, values = list(zip(*neighbor_dict.items()))
        node_indices = np.array(keys)
        means, sizes = np.array(values).T

        # log transform average
        if self.log:
            means = np.exp(means)

        # store outcome
        self.data.loc[node_indices, self.averaged_attr] = means
        self.data.loc[node_indices, self.size_attr] = sizes

    @default_figure
    def histogram_sample_sizes(self, ax=None):
        """ Histogram sample sizes. """
        sizes = self.data[self.size_attr].values
        _ = ax.hist(sizes, bins=np.arange(sizes.max()+1))

    @square_figure
    def plot_neigborhood(self, node, ax=None, **kwargs):
        """ Visualize neighborhood surrounding <node>. """
        neighbors = self.neighbors[node]
        colors = np.array(['k' for _ in range(self.graph.nodes.size)])
        colors[self.graph.position_map(node)] = 'g'
        colors[self.graph.position_map(neighbors)] = 'r'
        ax.scatter(*self.data[self.graph.xykey].values.T, c=colors, **kwargs)

    @default_figure
    def plot_autocorrelation(self, ax=None, xmax=10, **kwargs):
        """ Plot autocorrelation versus path length. """

        # evaluate distance between all nodes
        paths = dict(nx.all_pairs_shortest_path_length(self.G, cutoff=xmax))

        if xmax is None:
            max_depth = max([max(v.values()) for v in paths.values()])
        else:
            max_depth = xmax

        # get node levels and evaluate global mean/variance
        levels = self.node_values
        mu, sigma2 = levels.mean(), levels.var()

        def eval_flux(edge_list):
            """ Evaluate mean fluctuation in edge_list. """
            idx = self.graph.position_map(np.array(edge_list))
            flux = ((levels[idx[:, 0]]-mu) * (levels[idx[:, 1]] - mu)) / sigma2
            return flux.mean(), flux.std(), flux.size

        # compute pairwise fluctuations between nodes
        edges = {i: [] for i in range(max_depth+1)}
        for node, neighbors in paths.items():
            for neighbor, distance in neighbors.items():
                edges[distance].append((node, neighbor))

        # compile autocorrelation function
        means, _, sizes = list(zip(*[eval_flux(e) for e in edges.values()]))
        means, sizes = np.array(means), np.array(sizes)

        # plot autocorrelation
        ax.plot(range(max_depth+1), means, '.-k', **kwargs)
        ax.set_ylim(-0.1, 1)
        ax.set_xlim(0, max_depth+1)
        ax.set_ylabel('Correlation')
        ax.set_xlabel('Path length')


class CommunitySampler(NeighborSampler):
    """
    Class for sampling node attributes averaged over local community.

    Attributes:

        graph (spatial.Graph) - graph instance

        G (nx.Graph) - graph with node attribute

        attr (str) - attribute to be averaged over neighbors

        depth (int) - hierarchical level to which communities are merged

        log (bool) - if True, log-transform values before averaging

        twolevel (bool) - if True, use two-level community clustering

    """

    def __init__(self, graph, attr, depth=None, log=True, twolevel=False):
        """
        Instantiate sampler for averaging <attr> value over all neighbors within <depth> of each node.

        Args:

            graph (spatial.Graph) - graph instance

            attr (str) - attribute to be averaged over neighbors

            depth (int) - hierarchical level to which communities are merged

            twolevel (bool) - if True, use two-level community clustering

            log (bool) - if True, log-transform values before averaging

        """

        # store attributes
        self.graph = graph
        self.attr = attr
        self.depth = depth
        self.twolevel = twolevel
        self.log = log

        # perform community detection
        self.graph.find_communities(level=depth, twolevel=twolevel)

        # sample over neigbors
        self.average_over_neighbors()

    @property
    def neighbors(self):
        """ Dictionary of neighbor indices keyed by node indices. """
        gb = self.data.groupby('community')
        exclude = lambda node, neighbors: neighbors[neighbors!=node]
        neighbors = {n: exclude(n, gb.indices[c]) for n, c in dict(self.data.community).items()}
        return neighbors

    @property
    def size_attr(self):
        """ Neighborhood size attribute name. """
        return 'community_size'

    @property
    def averaged_attr(self):
        """ Name of averaged attribute. """
        return '{:s}_community'.format(self.attr)

    def average_over_neighbors(self):
        """ Average attribute value over all members of the community encompassing each node. """

        if self.log:
            agg = lambda x: np.log(x).mean()
        else:
            agg = lambda x: x.mean()

        # average over each community
        community_levels = self.data.groupby('community')[self.attr].aggregate(agg)
        community_to_mean_level = np.vectorize(dict(community_levels).get)
        means = community_to_mean_level(self.data.community.values)

        # log transform average
        if self.log:
            means = np.exp(means)

        # evaluate community sizes
        neighbors = self.neighbors
        get_community_size = np.vectorize(lambda n: len(neighbors[n]))

        # store outcome
        self.data[self.averaged_attr] = means
        self.data[self.size_attr] = get_community_size(self.data.index.values)

    @default_figure
    def plot_autocorrelation(self, ax=None):
        """ Plot autocorrelation versus community depth. """

        # construct dataframe
        data = deepcopy(self.data[['community']])
        data['levels'] = self.node_values
        data['zscore'] = (data.levels-data.levels.mean())/data.levels.std()

        # define functions for evaluation fluctuations
        f = lambda x: sum([sum([a * b for b in x if a!=b]) for a in x]) / 2
        g = lambda x: len(x)*(len(x)-1) / 2
        evaluate_mean_fluctuation = lambda key: data.groupby(key)['zscore'].agg(f).sum() / data.groupby(key)['zscore'].agg(g).sum()

        # instantiate infomap clustering
        detector = InfoMap(self.graph.edge_list)

        # evaluate autocorrelation function
        autocorrelation = []
        for level in range(detector.aggregator.max_depth):
            key = '{:d}'.format(level)
            data[key] = detector.aggregator(data.community, level=level)
            mean_fluctuation = evaluate_mean_fluctuation(key)
            autocorrelation.append((level, mean_fluctuation))

        # plot autocorrelation
        ax.plot(*list(zip(*autocorrelation)), '.-k')
        ax.set_ylim(-0.1, 1)
        ax.set_ylabel('Correlation')
        ax.set_xlabel('Hierarchical level')

    @default_figure
    def plot_autocorrelation_with_distance(self, ax=None):
        """ Plot autocorrelation versus community depth. """

        # construct dataframe
        data = deepcopy(self.data[['community']])
        data['levels'] = self.node_values

        # define functions for evaluation fluctuations
        f = lambda x: sum([sum([a * b for b in x if a!=b]) for a in x]) / 2
        g = lambda x: len(x)*(len(x)-1) / 2
        d = lambda x: np.mean([[np.sqrt((a-b)**2) for b in x if a != b] for a in x])

        def evaluate_mean_fluctuation(bin_id):
            total_fluctuations = data.groupby(bin_id)['zscore'].agg(f).sum()
            bin_size = data.groupby(bin_id)['zscore'].agg(g).sum()
            return total_fluctuations / bin_size

        def evaluate_mean_distance(bin_id):
            return data.groupby(bin_id)[self.graph.xykey].agg(d).mean()

        # instantiate infomap clustering
        detector = InfoMap(self.graph.edge_list)

        # evaluate autocorrelation function
        distances, autocorrelation = [], []
        for level in range(detector.aggregator.max_depth):
            key = '{:d}'.format(level)
            data[key] = detector.aggregator(data.community, level=level)
            mean_fluctuation = evaluate_mean_fluctuation(key)
            autocorrelation.append((level, mean_fluctuation))
            distances.append(evaluate_mean_distance(key))

        return distances, autocorrelation



class RadialSampler(NeighborSampler):
    """
    Class for sampling node attributes averaged within a predetermined radius of each node.

    Attributes:

        graph (spatial.Graph) - graph instance

        G (nx.Graph) - graph with node attribute

        attr (str) - attribute to be averaged over neighbors

        depth (int) - hierarchical level to which communities are merged

        log (bool) - if True, log-transform values before averaging

        twolevel (bool) - if True, use two-level community clustering

    """

    def __init__(self, graph, attr, depth=1., log=True):
        """
        Instantiate sampler for averaging <attr> value over all nodes within a predetermined radius of each node. The radius is defined by <depth> multiples of the characteristic length over which correlations in the attribute value decay.

        Args:

            graph (spatial.Graph) - graph instance

            attr (str) - attribute to be averaged over neighbors

            depth (int) - hierarchical level to which communities are merged

            log (bool) - if True, log-transform values before averaging

        """

        # store attributes
        self.graph = graph
        self.attr = attr
        self.depth = depth
        self.log = log

        # determine characteristic length scale and set sampling radius
        length_scale = graph.get_correlations(attr, log).characteristic_length

        # if failed (e.g. no correlation), use 1.5x median edge length
        if length_scale is None:
            length_scale = 1.5 * graph.median_edge_length

        self.length_scale = length_scale
        self.radius = depth * length_scale
        self.neighbor_mask = self._neighbor_mask

        # sample over neigbors
        self.average_over_neighbors()

    @property
    def distance_matrix(self):
        """ Euclidean distance matrix between nodes. """
        return self.graph.distance_matrix

    @property
    def _neighbor_mask(self):
        """ Boolean adjacency mask (True for neighbors). """
        neighbor_mask = (self.distance_matrix < self.radius)
        np.fill_diagonal(neighbor_mask, False)
        return neighbor_mask

    @property
    def neighbors(self):
        """ Dictionary of neighbor indices keyed by node indices. """
        return {n: r.nonzero()[0] for n, r in enumerate(self.neighbor_mask)}

    @property
    def size_attr(self):
        """ Neighborhood size attribute name. """
        return 'sampling_radius'

    @property
    def averaged_attr(self):
        """ Name of averaged attribute. """
        return '{:s}_radial'.format(self.attr)

    def average_over_neighbors(self):
        """
        Average attribute value over all nodes within the specified radius of each node.
        """

        # average within each neighborhood
        node_values = self.node_values.reshape(1, -1)
        matrix = np.repeat(node_values, self.num_nodes, axis=0)
        masked_values = np.ma.masked_array(matrix, mask=~self.neighbor_mask)
        means = masked_values.mean(axis=1).data

        # log transform average
        if self.log:
            means = np.exp(means)

        # store outcome
        self.data[self.averaged_attr] = means
        self.data[self.size_attr] = (~masked_values.mask).sum(axis=1)

    @square_figure
    def plot_neigborhood(self, node, ax=None, **kwargs):
        """ Visualize neighborhood surrounding <node>. """

        # draw selection boundary
        idx = self.graph.position_map(node)
        center = self.data[self.graph.xykey].values[idx]
        circle = plt.Circle(center, self.radius, color='r', alpha=0.2)
        ax.add_artist(circle)

        # scatter points
        super().plot_neigborhood(node, ax=ax, **kwargs)

    @default_figure
    def plot_autocorrelation(self, ax=None, **kwargs):
        """ Plot autocorrelation versus community depth. """
        correlations = self.graph.get_correlations(self.attr, self.log)
        correlations.visualize(ax=ax, **kwargs)



# """
# COMPUTE DISTANCE CORRELATIONS WITHIN COMMUNITY BINS:
# """

# from clones.annotation.spatial.sampling import CommunitySampler
# from copy import deepcopy
# import pandas as pd

# class Test(CommunitySampler):

#     @staticmethod
#     def evaluate_distances(xy):
#         """ Euclidean distance matrix between all nodes. """
#         x, y = xy.T
#         x = x.reshape(-1, 1)
#         y = y.reshape(-1, 1)
#         x_component = np.repeat(x**2, x.size, axis=1) + np.repeat(x.T**2, x.size, axis=0) - 2*np.dot(x, x.T)
#         y_component = np.repeat(y**2, y.size, axis=1) + np.repeat(y.T**2, y.size, axis=0) - 2*np.dot(y, y.T)
#         distance_matrix = np.sqrt(x_component + y_component)
#         return distance_matrix[np.triu_indices(len(distance_matrix), k=1)].tolist()

#     @staticmethod
#     def evaluate_fluctuations(values):
#         """ Euclidean distance matrix between all nodes. """
#         fluctuations_matrix = np.dot(values.reshape(-1, 1), values.reshape(1, -1))
#         return fluctuations_matrix[np.triu_indices(len(fluctuations_matrix), k=1)].tolist()

#     def plot_autocorrelation_with_distance(self, ax=None):
#         """ Plot autocorrelation versus community depth. """

#         # construct dataframe
#         data = deepcopy(self.data[['community']+self.graph.xykey])
#         data['levels'] = self.node_values
#         data['zscore'] = (data.levels-data.levels.mean())/data.levels.std()

#         def get_fluctuations(bin_id):
#             #f = lambda x: pd.Series({'fluctuations': self.evaluate_fluctuations(x.values)})
#             f = lambda x: self.evaluate_fluctuations(x.values)
#             fluctuations = []
#             for vector in data.groupby(bin_id)['zscore'].apply(f).values:
#                 fluctuations += vector
#             return fluctuations

#         def get_distances(bin_id):
#             d = lambda x: pd.Series({'distances': self.evaluate_distances(x.values)})
#             distances = []
#             for vector in data.groupby(bin_id)[self.graph.xykey].apply(d)['distances']:
#                 distances += vector
#             return distances

#         # instantiate infomap clustering
#         detector = InfoMap(self.graph.edge_list)

#         # evaluate autocorrelation function
#         distances, fluctuations = [], []
#         for level in range(detector.aggregator.max_depth):
#             bin_id = '{:d}'.format(level)
#             data[bin_id] = detector.aggregator(data.community, level=level)

#             distances.append(get_distances(bin_id))
#             fluctuations.append(get_fluctuations(bin_id))

#         return distances, fluctuations
