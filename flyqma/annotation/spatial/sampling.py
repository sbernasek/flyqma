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
    def plot_neigborhood(self, node,
                         color='r',
                         bg_color='k',
                         ax=None,
                         **kwargs):
        """
        Visualize neighborhood surrounding <node>.

        Args:

            node (int) - node positional index

            color (str or RGB tuple) - color for nodes within neighborhood

            bg_color (str or RGB tuple) - color for nodes outside neighborhood

            kwargs: keyword arguments for plt.scatter

        """
        node_id = self.graph.node_map(node)[0]
        neighbors_id = self.neighbors[node_id]
        neighbors = self.graph.position_map(neighbors_id)

        colors = np.array([bg_color for _ in range(self.graph.nodes.size)])
        colors[node] = color
        colors[neighbors] = color
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

        depth (int) - mean correlation lifetime

        level (int) - hierarchical level at which clusters are merged

        log (bool) - if True, log-transform values before averaging

        twolevel (bool) - if True, use two-level community clustering

    """

    def __init__(self, graph, attr, depth=1., log=True, twolevel=False):
        """
        Instantiate sampler for averaging <attr> value over all neighbors within <depth> of each node.

        Args:

            graph (spatial.Graph) - graph instance

            attr (str) - attribute to be averaged over neighbors

            depth (int) - mean correlation lifetime

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
        if self.graph.imap is None:
            self.graph.detect_communities(twolevel=twolevel)

        # determine clustering level
        self.level = self.clustering_level

        # sample over neigbors
        self.average_over_neighbors()

    @property
    def neighbors(self):
        """ Dictionary of neighbor indices keyed by node indices. """
        labels = self.graph._assign_community(self.level)
        gb = self.data.groupby(labels)
        exclude = lambda node, neighbors: neighbors[neighbors!=node]
        key = lambda n: self.data.index[n]
        return {key(n): exclude(key(n), gb.indices[c]) for n,c in enumerate(labels)}

    @property
    def size_attr(self):
        """ Neighborhood size attribute name. """
        return 'community_size'

    @property
    def averaged_attr(self):
        """ Name of averaged attribute. """
        return '{:s}_community'.format(self.attr)

    @property
    def z_attr(self):
        """ Name of z-scored attribute. """
        return self.attr+'_zscore'

    @property
    def clustering_level(self):
        """ Highest clustering level at which the mean correlation remains above <self.depth> multiples of the decay constant. """
        _, correlations = self.autocorrelate(include_distances=False)
        correlations = np.array(correlations)
        relative_correlation = correlations/correlations[0]
        above = relative_correlation >= np.exp(-self.depth)
        return above.nonzero()[0][-1]

    def average_over_neighbors(self):
        """ Average attribute value over all members of the community encompassing each node. """

        if self.log:
            agg = lambda x: np.log(x).mean()
        else:
            agg = lambda x: x.mean()

        # average over each community
        labels = self.graph._assign_community(self.level)
        community_levels = self.data.groupby(labels)[self.attr].aggregate(agg)
        community_to_mean_level = np.vectorize(dict(community_levels).get)
        means = community_to_mean_level(labels)

        # log transform average
        if self.log:
            means = np.exp(means)

        # evaluate community sizes
        neighbors = self.neighbors
        get_community_size = np.vectorize(lambda n: len(neighbors[n]))

        # store outcome
        self.data[self.averaged_attr] = means
        self.data[self.size_attr] = get_community_size(self.data.index.values)

    def autocorrelate(self, include_distances=False):
        """
        Returns autocorrelation versus community depth.

        Args:

            include_distances (bool) - return mean separate distances

        Returns:

            levels (list) - clustering depths, starting from finest resolution

            correlations (list) - mean correlation within communities

            <optional> distances (list) - mean pairwise separation distance

        """

        # compile z-scored node values
        node_v = self.node_values
        self.data[self.z_attr] = (node_v-node_v.mean())/node_v.std()

        # define hierarchical levels
        levels = list(range(self.graph.imap.aggregator.max_depth))

        # define aggregation functions
        def get_binsize(x):
            """ Evaluate total number of node pairs. """
            return x.size * (x.size - 1) / 2

        def get_fluctuations(x):
            """ Evaluate total pairwise fluctuation between nodes. """
            v = x.values.reshape(-1, 1)
            return self.graph.get_matrix_upper(np.dot(v, v.T)).sum()

        def get_distances(x):
            """ Evaluate total pairwise distance between nodes. """
            distance_matrix = self.graph._distance_matrix(x.values)
            distances = self.graph.get_matrix_upper(distance_matrix)
            return distances.sum()

        # evaluate autocorrelation at each hierarchical level
        correlations, distances = [], []
        for level in levels:
            labels = self.graph._assign_community(level)
            gb = self.data.groupby(labels)
            total_flux = gb[self.z_attr].agg(get_fluctuations).sum()
            num_of_flux = gb[self.z_attr].agg(get_binsize).sum()
            correlations.append(total_flux / num_of_flux)

            if include_distances:
                total_distance = gb[self.graph.xykey].apply(get_distances).sum()
                distances.append(total_distance/num_of_flux)

        if include_distances:
            return distances, correlations
        else:
            return levels, correlations

    @default_figure
    def plot_autocorrelation(self, ax=None, spatial=False):
        """
        Plot autocorrelation versus community depth.

        Args:

            distance (bool) - if True, plot spatial correlation

        """


        levels, correlation = self.autocorrelate(include_distances=spatial)

        # plot autocorrelation
        ax.plot(levels, correlation, '.-k')
        ax.set_ylim(-0.1, 1)
        ax.set_ylabel('Correlation')

        # format axis
        if spatial:
            ax.set_xlabel('Distance')
        else:
            ax.set_xlabel('Hierarchical level')


class RadialSampler(NeighborSampler):
    """
    Class for sampling node attributes averaged within a predetermined radius of each node.

    Attributes:

        graph (spatial.Graph) - graph instance

        G (nx.Graph) - graph with node attribute

        attr (str) - attribute to be averaged over neighbors

        depth (int) - hierarchical level to which communities are merged

        log (bool) - if True, log-transform values before averaging

        length_scale (float) - characteristic length scale of the graph

        radius (float) - radius of sampling region surrounding each measurement

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
        """ Euclidean distance matrix between nodes (ordered by position in <self.data>). """
        return self.graph.distance_matrix

    @property
    def _neighbor_mask(self):
        """ Boolean adjacency mask (True for neighbors). """
        neighbor_mask = (self.distance_matrix < self.radius)
        np.fill_diagonal(neighbor_mask, False)
        return neighbor_mask

    @property
    def neighbors(self):
        """
        Dictionary of neighbor positional indices keyed by node indices.
        """
        return  {n: r.nonzero()[0] for n, r in enumerate(self.neighbor_mask)}

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
    def plot_neigborhood(self, node,
                         color='r',
                         bg_color='k',
                         radius=True,
                         ax=None,
                         **kwargs):
        """
        Visualize neighborhood surrounding <node>.

        Args:

            node (int) - node positional index

            color (str or RGB tuple) - color for nodes within neighborhood

            bg_color (str or RGB tuple) - color for nodes outside neighborhood

            radius (bool) - if True, plot sampling radius

            kwargs: keyword arguments for plt.scatter

        """

        # draw selection boundary
        if radius:
            center = self.data[self.graph.xykey].values[node]
            circle = plt.Circle(center, self.radius, color=color, alpha=0.2)
            ax.add_artist(circle)

        # scatter points
        colors = np.array([bg_color for _ in range(self.graph.nodes.size)])
        colors[node] = color
        colors[self.neighbors[node]] = color
        ax.scatter(*self.data[self.graph.xykey].values.T, c=colors, **kwargs)

    @default_figure
    def plot_autocorrelation(self, ax=None, **kwargs):
        """ Plot autocorrelation versus community depth. """
        correlations = self.graph.get_correlations(self.attr, self.log)
        correlations.visualize(ax=ax, **kwargs)
