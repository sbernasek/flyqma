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
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

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
        return self.community_classifier(cells.community)


class Tessellation:

    def __init__(self, xy, labels, q=90, colors=None):

        self.vor = Voronoi(xy)
        self.vor.regions = np.array(self.vor.regions)
        self.set_region_mask(q=q)

        self.region_labels = self.label_regions(labels)

        self.verts = self.vor.regions[self.mask]


        #self.labels = labels
        self.set_cmap(colors)

    def label_regions(self, labels):
        #region_to_point = np.vectorize({r: p for p, r in enumerate(self.vor.point_region)}.get)
        points = np.argsort(self.vor.point_region)
        point_to_label = np.vectorize(dict(enumerate(labels)).get)
        region_labels = point_to_label(points)
        return region_labels

    def set_cmap(self, colors=None):
        N = len(set(self.region_labels))
        if colors is None:
            colors = np.random.random((N, 3))
        self.cmap = ListedColormap(colors, 'indexed', N)

    @staticmethod
    def _evaluate_area(x, y):
        """ Compute area enclosed by a set of points. """
        return 0.5*np.abs(np.dot(x, np.roll(y,1))-np.dot(y, np.roll(x,1)))

    def evaluate_region_area(self, region):
        return self._evaluate_area(*self.vor.vertices[region, :].T)

    def set_region_mask(self, q=90):
        f = np.vectorize(lambda x: -1 not in x and len(x) > 0)
        mask = f(self.vor.regions)
        mask *= self.build_region_area_mask(q=q)
        self.mask = mask

    def build_region_area_mask(self, q=90):
        evaluate_area = np.vectorize(lambda x: self.evaluate_region_area(x))
        areas = evaluate_area(self.vor.regions)
        threshold = np.percentile(areas, q=q)
        return (areas <= threshold)

    @staticmethod
    def _show(vertices, c='k', ax=None, alpha=0.5):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlim(0, 2048)
            ax.set_ylim(0, 2048)
            ax.axis('off')
        poly = PolyCollection(vertices)
        poly.set_facecolors(c)
        poly.set_alpha(alpha)
        ax.add_collection(poly)


    def show(self, ax=None, **kw):
        get_vertices = np.vectorize(lambda region: self.vor.vertices[region])
        #vertices = get_vertices(self.vor.regions[self.mask])
        vertices = [self.vor.vertices[r] for r in self.vor.regions[self.mask]]

        c = self.cmap(self.region_labels[self.mask[1:]])
        self._show(vertices, c=c, ax=ax, **kw)


class CloneVisualization(Tessellation):

    def __init__(self, df, label='genotype', **kw):
        xy = df[['centroid_x', 'centroid_y']].values
        labels = df[label].values
        Tessellation.__init__(self, xy, labels, **kw)



class Labeler:
    """ Label cells on specified quantity. """

    def __init__(self, label_on='genotype'):
        labels = {0:'m', 1:'h', 2:'w', -1:'none'}
        self.labeler = np.vectorize(labels.get)
        self.label_on = label_on

    def __call__(self, cells):
        return self.labeler(cells[self.label_on])


class Concurrency:
    """ Determines minimum x-distance to each cell type. """

    def __init__(self, cells, basis='cell_type', min_pop=5, tolerance=10):
        self.cells = cells
        self.basis = basis
        self.unique_labels = self.cells[self.basis].unique()
        self.min_pop = min_pop
        self.tolerance = tolerance

    def evaluate_distance(self, target):
        candidates = self.cells[self.cells[self.basis]==target]
        if len(candidates) > self.min_pop:
            rs = lambda x: x.centroid_x.values.reshape(-1, 1)
            distances = cdist(rs(self.cells), rs(candidates)).min(axis=1)
        else:
            distances = 1000*np.ones(len(self.cells), dtype=np.float64)
        return distances

    def assign_concurrency(self):
        """ Assign concurrency for all unique labels. """
        for label in self.unique_labels:
            distances = self.evaluate_distance(label)
            self.cells['concurrent_'+label] = (distances <= self.tolerance)





