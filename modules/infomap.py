import numpy as np
import infomap


class InfoMap:

    def __init__(self, graph, weighted=True, channel='r', **kw):
        self.weighted = weighted
        self.channel = channel
        self.infomap = self.build_network(graph, weighted, channel, **kw)
        self.run()
        self.classifier = self.build_classifier()

    def __call__(self, x):
        return self.classifier(x)

    @staticmethod
    def build_network(graph, weighted=True, channel='r', infomap_args=''):
        """ Compile InfoMap object from graph. """

        # instantiate infomap
        infomap_obj = infomap.Infomap("--two-level --undirected")
        network = infomap_obj.network()

        # add edges
        links = graph.build_links(weighted=weighted, channel=channel)
        _ = [network.addLink(*e) for e in links]

        return infomap_obj

    def run(self, report=False):
        self.infomap.run()
        if report:
            print("Identified {:d} modules.".format(self.infomap.numTopModules()))

    def build_classifier(self):
        classes = {}
        for node in self.infomap.iterTree():
            if node.isLeaf():
                classes[node.physicalId] = node.moduleIndex()
        return np.vectorize(classes.get)
