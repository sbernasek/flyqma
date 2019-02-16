import numpy as np
import networkx as nx
from collections import Counter

from .labelers import AttributeLabeler


def __call__(self, communities):
        """ Classify communities. """
        return self.classifier(communities)

    @classmethod
    def from_layer(cls, layer, cell_classifier, **kwargs):
        """ Instantiate community classifier from a layer. """
        return cls.__class__(layer.data, cell_classifier, **kwargs)


class Annotation(AttributeLabeler):
    """
    Object for assigning labels to measurements.

    Attributes:

        graph (Graph) - graph connecting adjacent cells

        cell_classifier (CellClassifier) - callable object

        labeler (InfomapClassifier) - callable object

    Inherited attributes:

        label (str) - name of label field to be added

        attribute (str) - existing cell attribute used to determine labels

    """

    def __init__(self, graph, cell_classifier,
                 label='community_genotype',
                 attribute='community',
                 twolevel=False,
                 **kwargs):
        """
        Instantiate community-based genotype annotation object.

        Args:

            graph (Graph) - graph connecting adjacent cells

            cell_classifier (CellClassifier) - callable object

            label (str) - name of <genotype> attribute to be added

            attribute (str) - name of attribute defining community affiliation

            twolevel (bool) - if True, perform two-level clustering

            kwargs: keyword arguments for InfomapClassifier

        """

        # store label and attribute field names
        self.label = label
        self.attribute = attribute

        # run community detection and store graph
        graph.find_communities(twolevel=twolevel)
        self.graph = graph

        # store cell classifier
        self.cell_classifier = cell_classifier

        # build genotype labeler based on community classifier
        self.labeler = self.build_classifier(**kwargs)

    @classmethod
    def from_layer(cls, layer, **kwargs):
        """
        Instantiate from layer.

        Args:

            layer (Layer)

        Returns:

            labeler (InfomapLabeler derivative)

        """
        return cls.__class__(layer.graph, layer.classifier, **kwargs)

    def __call__(self, communities):
        """ Classify communities. """
        return self.classifier(communities)

    @classmethod
    def from_layer(cls, layer, cell_classifier, **kwargs):
        """ Instantiate community classifier from a layer. """
        return cls.__class__(layer.data, cell_classifier, **kwargs)

    def build_classifier(self, **kwargs):
        """
        Build community classifier.

        Args:

            kwargs: keyword arguments for community classifier

        Returns:

            classifier (func) - maps communities to labels

        """

        # assign community labels
        self.graph.df['community'] = -1
        ind = self.graph.nodes
        self.graph.df.loc[ind, 'community'] = self.graph.community_labels

        # build community classifier
        classifier = InfomapClassifier(self.graph.df,
                                         self.cell_classifier,
                                         **kwargs)

        return classifier


class KatzLabeler(InfomapLabeler):
    """
    Object for assigning genotypes to cells based on their local community by using the Katz centrality of the posterior distributions fit to cell fluorescence levels.

    Attributes:

        graph (Graph) - graph connecting adjacent cells

        cell_classifier (CellClassifier) - callable object

        labeler (KatzClassifier) - callable object

    Inherited attributes:

        label (str) - name of label field to be added

        attribute (str) - existing cell attribute used to determine labels

    """

    def __init__(self, graph, cell_classifier,
                 label='katz_genotype',
                 **kwargs):
        """
        Instantiate community-based genotype annotation object.

        Args:

            graph (Graph) - graph connecting adjacent cells

            cell_classifier (CellClassifier) - callable object

            label (str) - name of <genotype> attribute to be added

            kwargs: keyword arguments for KatzClassifier, e.g. alpha

        """

        # store label and attribute field names
        self.label = label

        # run community detection and store graph
        self.graph = graph

        # store cell classifier
        self.cell_classifier = cell_classifier

        # build genotype labeler based on community classifier
        self.labeler = self.build_classifier(**kwargs)

    def assign_labels(self, data):
        """
        Assign labels by adding <label> field to cell measurement data.

        Args:

            data (pd.DataFrame) - cells measurement data

        """
        data[self.label] = self.labeler(data.index.values)

    def build_classifier(self, **kwargs):
        """
        Build Katz classifier.

        Args:

            kwargs: keyword arguments for Katz classifier, e.g. alpha

        Returns:

            classifier (func) - maps measurement index to cell genotype

        """
        return self._build_classifier(self.graph, self.cell_classifier, **kwargs)

    @staticmethod
    def _build_classifier(graph, cell_classifier, alpha=0.9):
        """
        Construct classifier baed on the maximum Katz centrality of posterior distributions applied to undirected edges weighted by node similarity.

        Args:

            graph (Graph) - graph connecting adjacent cells

            cell_classifier (BayesianClassifier) - labels individual cells

            alpha (float) - attenuation factor

        Returns:

            genotypes_dict (dict) - maps measurement index to genotype

        """

        # build undirected graph weighted by node similarity
        G = graph.get_networkx()

        # evaluate posterior genotype distribution for each node
        posterior = cell_classifier.evaluate_posterior(graph.df.loc[list(G.nodes)])

        # compile normalized adjacency matrix
        adjacency = nx.to_numpy_array(G)
        adjacency /= adjacency.sum(axis=0)

        # evaluate centrality
        n = np.array(adjacency).shape[0]
        centrality = np.linalg.solve(np.eye(n, n)-(alpha*adjacency), (1-alpha)*posterior)

        # build classifier that maps model distributions to genotypes.
        #get_label = np.vectorize(cell_classifier.component_to_label.get)
        node_labels = centrality.argmax(axis=1)

        # return genotype mapping
        index_to_genotype = dict(zip(list(G.nodes), node_labels))

        return np.vectorize(index_to_genotype.get)
