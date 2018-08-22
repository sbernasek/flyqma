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
from modules.classification import CommunityClassifier



# class CloneMask:

#     def __init__(self, graph, genotypes):

#         # label unmasked triangles with genotype
#         exclusion = graph.tri.mask
#         genotype_labeler = np.vectorize({n:g for n, g in zip(graph.nodes, genotypes)}.get)
#         nodes = graph.node_map(graph.tri.triangles[~exclusion])
#         node_genotypes = genotype_labeler(nodes)

#         # define triangle genotypes
#         borders = self.get_borders(node_genotypes)
#         genotypes = node_genotypes[:, 0]
#         genotypes[borders] = -1
#         self.genotypes = genotypes # np.ma.masked_array(genotypes, borders)

#         N = graph.tri.mask.size
#         mask = np.ma.masked_where(exclusion, np.ones(N, dtype=int))
#         mask[~exclusion] = self.genotypes
#         self.mask = mask

#     @staticmethod
#     def from_layer(layer):
#         graph = layer.annotation.graph
#         genotypes = layer.df.loc[graph.nodes].genotype
#         return CloneMask(graph, genotypes)

#     @staticmethod
#     def get_borders(x):
#         return (x.max(axis=1) != x.min(axis=1))



class Annotation:

    def __init__(self, graph, cell_classifier):

        # run community detection and store graph
        graph.find_communities()
        self.graph = graph

        # store cell classifier
        self.cell_classifier = cell_classifier
        self.community_classifier = self.build_classifier()


        # if 'genotype' in df.columns.unique():
        #     clone_mask = CloneMask(self.graph, df.loc[self.graph.nodes].genotype)
        #     self.clone_mask = clone_mask

        # self.set_colormap()

    def __call__(self, cells):
        """
        Annotate cells using cell classifier.

        Args:
        cells (pd.DataFrame) - cells to be classified

        Returns:
        labels (pd.Series) - classifier output
        """
        return self.annotate(cells)

    def build_classifier(self):
        """
        Build community classifier.

        Returns:
        classifier (func) - maps communities to labels
        """

        # assign community labels
        self.graph.df['community'] = -1
        ind = self.graph.nodes
        self.graph.df.loc[ind, 'community'] = self.graph.community_labels

        # build community classifier
        classifier = CommunityClassifier(self.graph.df, self.cell_classifier)

        return classifier

    def annotate(self, cells):
        """
        Annotate cells using cell classifier.

        Args:
        cells (pd.DataFrame) - cells to be classified

        Returns:
        labels (pd.Series) - classifier output
        """
        return self.community_classifier(cells)





    # def __call__(self, ind):
    #     """ Assign genotype t """
    #     nodes = self.graph.nodes
    #     genotypes = self.clustering.genotypes
    #     node_to_genotype = dict(zip(nodes, genotypes))

    #     # add nodes excluded by the triangulation distance filter
    #     excluded = self.graph.excluded
    #     node_to_genotype.update(dict(zip(excluded, -1*np.ones_like(excluded))))

    #     return np.vectorize(node_to_genotype.get)(ind)

    # def set_colormap(self, colors=None, border_color='y'):
    #     """ Set colormap for clones. """
    #     if colors is None:
    #         colors = 'rgb'
    #     cmap = ListedColormap(colors)
    #     cmap.set_under(border_color)
    #     self.cmap = cmap
    #     self.norm = Normalize(vmin=0, vmax=2)

    # def plot_clones(self, ax, alpha=0.2, **kw):
    #     """ Add triangulation colored by genotype. """
    #     self.graph.plot_triangles(ax=ax,
    #                               colors=self.clone_mask.mask,
    #                               cmap=self.cmap,
    #                               norm=self.norm,
    #                               alpha=alpha, **kw)

