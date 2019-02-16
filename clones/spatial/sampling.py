import numpy as np
import networkx as nx
from ..annotation.classification import BivariateMixtureClassifier
from ..visualization.settings import default_figure


class SpatialSampler:
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

        self.build_graph()
        self.average_over_neighbors()

    @property
    def df(self):
        """ Graph data. """
        return self.graph.df

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

    def build_graph(self):
        """ Build networkx graph object. """

        # log transform attribute
        if self.log:
            self.df[self.attr_used] = np.log(self.df[self.attr].values)

        # construct networkx object
        self.G = self.graph.get_networkx(self.attr_used)

    @staticmethod
    def _average_over_neighbors(G, attribute, depth=1):
        """
        Average attribute value over all neighbors of each node in a graph.

        Args:

            G (nx.Graph) - graph with nodes containing an attribute

            attribute (str) - node attribute to be averaged over neighbors

            depth (int) - maximum number of edges connecting neighbors

        Returns:

            node_to_neighborhood (dict) - dictionary of neighborhood (<attribute>, size) tuples keyed by positional indices

        """

        # define dictionary related nodes to their attributes
        node_to_attr = nx.get_node_attributes(G, attribute)

        # define breadth first search of each <src> node
        bfs = lambda n: [e[1] for e in nx.bfs_edges(G, n, depth_limit=depth)]

        # define function to parse mean node attribute within neighborhood
        parse = lambda nbs: (np.mean([node_to_attr[n] for n in nbs]), len(nbs))

        return {src: parse(bfs(src)) for src in G.nodes}

    def average_over_neighbors(self):
        """
        Average <attr> value over all neighbors within <depth> of each node.

        """

        # define dictionary related nodes to their attributes
        node_to_attr = nx.get_node_attributes(self.G, self.attr_used)

        # average attribute over neighbors
        neighbor_dict = self._average_over_neighbors(self.G, self.attr_used, self.depth)

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

    def train_classifier(self, num_components=3, num_labels=3, **kwargs):
        """
        Train bivariate mixture model classifier on sample.

        Args:

            num_components (str) - number of mixture components

            num_labels (int) - number of unique labels

        Returns:

            model (BivariateMixtureClassifier)

        """

        kw = dict(classify_on=self.keys,
                      num_components=num_components,
                      num_labels=num_labels)
        kw.update(kwargs)

        return BivariateMixtureClassifier(self.sample,  **kw)

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
