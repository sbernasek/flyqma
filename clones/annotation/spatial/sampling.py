import numpy as np
import networkx as nx

from ...visualization.settings import *


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
    def df(self):
        """ Graph data. """
        return self.graph.df

    @property
    def G(self):
        """ NetworkX graph instance. """
        return self.graph.G

    @property
    def node_values(self):
        """ Dictionary of attribute values, keyed by node index. """
        values = self.df[self.attr]
        if self.log:
            values = np.log(values)
        return dict(values)

    @property
    def neighbors(self):
        """ Dictionary of neighbor indices keyed by node indices. """
        kwargs = dict(depth_limit=self.depth)
        bfs = lambda n: [e[1] for e in nx.bfs_edges(self.G, n, **kwargs)]
        return {node: bfs(node) for node in G.nodes}

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
        return self.df[self.keys].values

    def add_attribute_to_graph(self):
        """ Add attribute to networkx graph object. """
        nx.set_node_attributes(self.G, self.node_values, name=self.attr_used)

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
        neighbor_dict = self._neighbor_average(self.G, self.node_values, self.depth)

        # extract average and sample size for each node
        keys, values = list(zip(*neighbor_dict.items()))
        node_indices = np.array(keys)
        means, sizes = np.array(values).T

        # log transform average
        if self.log:
            means = np.exp(means)

        # store outcome
        self.df.loc[node_indices, self.averaged_attr] = means
        self.df.loc[node_indices, self.size_attr] = sizes

    @default_figure
    def histogram_sample_sizes(self, ax=None):
        """ Histogram sample sizes. """
        sizes = self.df[self.size_attr].values
        _ = ax.hist(sizes, bins=np.arange(sizes.max()+1))

    @default_figure
    def plot_neigborhood(self, node, ax=None, figsize=(2, 2)):
        """ Visualize neighborhood surrounding <node>. """
        neighbors = self.neighbors[node]
        colors = np.array(['k' for _ in range(self.graph.nodes.size)])
        colors[self.graph.position_map(node)] = 'g'
        colors[self.graph.position_map(neighbors)] = 'r'
        ax.scatter(*self.graph.df[['x','y']].values.T, s=2, c=colors)


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
        gb = self.df.groupby('community')
        exclude = lambda node, neighbors: neighbors[neighbors!=node]
        neighbors = {n: exclude(n, gb.indices[c]) for n, c in dict(self.df.community).items()}
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
        community_levels = self.df.groupby('community')[self.attr].aggregate(agg)
        community_to_mean_level = np.vectorize(dict(community_levels).get)
        means = community_to_mean_level(self.df.community.values)

        # log transform average
        if self.log:
            means = np.exp(means)

        # evaluate community sizes
        neighbors = self.neighbors
        get_community_size = np.vectorize(lambda n: len(neighbors[n]))

        # store outcome
        self.df[self.averaged_attr] = means
        self.df[self.size_attr] = get_community_size(self.df.index.values)
