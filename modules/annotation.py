"""
TO DO:

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize

from sklearn.cluster import KMeans
from collections import Counter

from modules.graphs import WeightedGraph
from modules.infomap import InfoMap


class Clustering:

    def __init__(self, graph, weighted=True, channel='r', upper_bound=100, log=True):
        self.levels = graph.df[channel].loc[graph.nodes].values
        self.im = InfoMap(graph, weighted=weighted, channel=channel)
        self.im_labels = self.im(graph.nodes)
        self.mean_cluster_levels = self.evaluate_mean_cluster_levels()
        self.define_genotypes(graph, n_clusters=3, upper_bound=upper_bound, log=log)
        self.genotypes = self.assign_genotype_to_clusters()

    def evaluate_mean_cluster_levels(self):
        adict = dict(labels=self.im_labels, values=self.levels)
        return pd.DataFrame(adict).groupby('labels').mean()['values'].values

    def infomap_label_distribution(self):
        fig, ax = plt.subplots(figsize=(2, 1))
        _ = ax.hist(self.im_labels, bins=np.arange(0, self.im_labels.max()+1))

    def define_genotypes(self, graph, n_clusters=3, upper_bound=100, log=True):
        """ Cluster nodes by level. """

        params = dict(init='k-means++', n_init=10) #random_state=0

        # exclude points above threshold
        included = self.levels <= np.percentile(self.levels, q=upper_bound)
        levels = self.levels[included].reshape(-1, 1)

        # perform clustering
        if log:
            km = KMeans(n_clusters=n_clusters, **params).fit(np.log10(levels))
        else:
            km = KMeans(n_clusters=n_clusters, **params).fit(levels)

        # assemble labels
        self.kmeans_labels = np.zeros_like(included, dtype=int)
        self.kmeans_labels[included] = km.labels_
        self.kmeans_labels[~included] = np.argmax(km.cluster_centers_)

#        self.kmeans_labels = km.labels_
        centroids = km.cluster_centers_.ravel()

        # create vectorized function for mapping 0/1/2 genotypes to k_means labels
        flip = lambda f: f.__class__(map(reversed, f.items()))
        kmeans_to_genotype = flip(dict(enumerate(np.argsort(centroids))))
        self.kmeans_to_genotype = np.vectorize(kmeans_to_genotype.get)

    @staticmethod
    def get_mode(x):
        mode, count = Counter(x).most_common(1)[0]
        return mode

    def assign_genotype_to_clusters(self):
        get_genotypes = lambda x: self.kmeans_to_genotype(self.kmeans_labels[np.where(self.im_labels==x)])
        dominant_genotype = lambda x: self.get_mode(get_genotypes(x))
        im_to_genotype = {l: dominant_genotype(l) for l in range(self.im_labels.max()+1)}
        genotypes = np.vectorize(im_to_genotype.get)(self.im_labels)
        return genotypes


class CloneMask:

    def __init__(self, graph, genotypes):

        # label unmasked triangles with genotype
        exclusion = graph.tri.mask
        genotype_labeler = np.vectorize({n:g for n, g in zip(graph.nodes, genotypes)}.get)
        nodes = graph.node_map(graph.tri.triangles[~exclusion])
        node_genotypes = genotype_labeler(nodes)

        # define triangle genotypes
        borders = self.get_borders(node_genotypes)
        genotypes = node_genotypes[:, 0]
        genotypes[borders] = -1
        self.genotypes = genotypes # np.ma.masked_array(genotypes, borders)

        N = graph.tri.mask.size
        mask = np.ma.masked_where(exclusion, np.ones(N, dtype=int))
        mask[~exclusion] = self.genotypes
        self.mask = mask

    @staticmethod
    def get_borders(x):
        return (x.max(axis=1) != x.min(axis=1))


class Annotation:

    def __init__(self, df,
                 q=95,
                 weighted=True,
                 channel='r',
                 fg_only=False,
                 upper_bound=100,
                 log=True):

        self.channel = channel

        # extract foreground (optional)
        self.fg_only = fg_only
        if fg_only:
            self.bg = df[~df.foreground].index
            df = df[df.foreground]
        else:
            self.bg = None

        # compile graph and get dataframe of distance-filtered nodes
        self.graph = WeightedGraph(df, q=q)

        # cluster graph and assign clone mask
        self.clustering = Clustering(self.graph, weighted=weighted, channel=channel, upper_bound=upper_bound, log=log)
        self.clone_mask = CloneMask(self.graph, self.clustering.genotypes)
        self.set_colormap()

    def __call__(self, ind):
        nodes = self.graph.nodes
        genotypes = self.clustering.genotypes
        node_to_genotype = dict(zip(nodes, genotypes))

        # add nodes excluded by the triangulation distance filter
        excluded = self.graph.excluded
        node_to_genotype.update(dict(zip(excluded, -1*np.ones_like(excluded))))

        # add nodes excluded from the foreground
        if self.bg is not None:
            bg = self.bg.values
            node_to_genotype.update(dict(zip(bg, -np.ones_like(bg))))

        return np.vectorize(node_to_genotype.get)(ind)

    def set_colormap(self, colors=None, border_color='y'):
        """ Set colormap for clones. """

        if colors is None:
            colors = 'rgb'
        cmap = ListedColormap(colors)
        cmap.set_under(border_color)
        self.cmap = cmap
        self.norm = Normalize(vmin=0, vmax=2)

    def plot_clones(self, ax, alpha=0.2, **kw):
        """ Add triangulation colored by genotype. """
        self.graph.plot_triangles(ax=ax,
                                  colors=self.clone_mask.mask,
                                  cmap=self.cmap,
                                  norm=self.norm,
                                  alpha=alpha, **kw)

    def slice_nodes_by_genotype(self, genotype):
        """ Returns array of nodes labeled with the specified genotype. """
        nodes = self.graph.node_map((self.clustering.genotypes==genotype).nonzero()[0])
        return nodes

    def slice_graph_by_genotype(self, genotype):
        """ Returns subgraph of nodes labeled with the specified genotype. """
        nodes = self.slice_nodes_by_genotype(genotype)
        return self.graph.get_subgraph(nodes)










