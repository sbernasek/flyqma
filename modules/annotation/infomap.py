import numpy as np
import infomap


class InfoMap:

    def __init__(self, edges, **kw):
        self.infomap = self.build_network(edges, **kw)
        self.run()
        node_to_module, classifier = self.build_classifier()
        self.node_to_module = node_to_module
        self.classifier = classifier

    def __call__(self, x):
        return self.classifier(x)

    @staticmethod
    def build_network(edges, infomap_args=''):
        """ Compile InfoMap object from graph edges. """

        # instantiate infomap
        infomap_obj = infomap.Infomap("--two-level --undirected")
        network = infomap_obj.network()

        # add edges
        _ = [network.addLink(*e) for e in edges]

        return infomap_obj

    def run(self, report=False):
        self.infomap.run()
        if report:
            print("Identified {:d} modules.".format(self.infomap.numTopModules()))

    def build_classifier(self):
        node_to_module = {}
        for node in self.infomap.iterTree():
            if node.isLeaf():
                node_to_module[node.physicalId] = node.moduleIndex()
        return node_to_module, np.vectorize(node_to_module.get)
