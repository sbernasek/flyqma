import numpy as np
import infomap


class InfoMap:
    """
    Object for performing infomap flow-based community detection.

    Attributes:

        infomap (infomap.Infomap) - infomap object

        node_to_module (dict) - {node: module} pairs

        classifier (vectorized func) - maps nodes to modules

    """

    def __init__(self, edges):
        """
        Instantiate infomap community detection. Two-level community detection is used by default.

        Args:

            edges (list) - (i, j, weight) tuple for each edge

        """

        self.infomap = self.build_network(edges)
        self.run()
        node_to_module, classifier = self.build_classifier()
        self.node_to_module = node_to_module
        self.classifier = classifier

    def __call__(self, x):
        """ Returns predicted class labels for values. """
        return self.classifier(x)

    @staticmethod
    def build_network(edges):
        """  Compile InfoMap object from graph edges. """

        # instantiate infomap
        infomap_obj = infomap.Infomap("--two-level --undirected")
        network = infomap_obj.network()

        # add edges
        _ = [network.addLink(*e) for e in edges]

        return infomap_obj

    def run(self, report=False):
        """
        Run infomap community detection.

        Args:

            report (bool) - if True, print number of modules found

        """
        self.infomap.run()
        if report:
            print("Found {:d} modules.".format(self.infomap.numTopModules()))

    def build_classifier(self):
        """
        Construct node to module classifier.

        Returns:

            node_to_module (dict) - {node: module} pairs

            classifier (vectorized func) - maps nodes to modules

        """
        node_to_module = {}
        for node in self.infomap.iterTree():
            if node.isLeaf():
                node_to_module[node.physicalId] = node.moduleIndex()
        return node_to_module, np.vectorize(node_to_module.get)
