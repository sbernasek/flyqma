import numpy as np
import infomap


class InfoMap:
    """
    Object for performing infomap flow-based community detection.

    Attributes:

        infomap (infomap.Infomap) - infomap object

        node_to_module (dict) - {node: module} pairs

        classifier (vectorized func) - maps nodes to modules

        aggregator (CommunityAggregator)

    """

    def __init__(self, edges, **kwargs):
        """
        Instantiate infomap community detection. Two-level community detection is used by default.

        Args:

            edges (list) - (i, j, weight) tuple for each edge

            kwargs: keyword arguments for build_network method, including:

                twolevel (bool) - if True, perform two-level clustering

                N (int) - number of trials

        """

        self.infomap = self.build_network(edges, **kwargs)
        self.run()
        node_to_module, classifier = self.build_classifier()
        self.node_to_module = node_to_module
        self.classifier = classifier
        self.aggregator = CommunityAggregator(self.infomap)

    def __call__(self, x, level=None):
        """ Returns predicted class labels for values. """
        return self.aggregator(self.classifier(x), level)

    @property
    def max_depth(self):
        """ Maximum tree depth. """
        return self.infomap.maxTreeDepth()

    @staticmethod
    def build_network(edges, twolevel=False, N=25):
        """
        Compile InfoMap object from graph edges.

        Args:

            twolevel (bool) - if True, perform two-level clustering

            N (int) - number of trials

        """

        # define arguments
        args = '--undirected --silent -N {:d}'.format(N)
        if twolevel:
            args = '--two-level ' + args

        # instantiate infomap
        infomap_obj = infomap.Infomap(args)
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
        for node in self.infomap.iterLeafNodes():
            node_to_module[node.physicalId] = node.moduleIndex()
        return node_to_module, np.vectorize(node_to_module.get)


class CommunityAggregator:
    """ Tool for hierarchical aggregation of communities. """

    def __init__(self, infomap):
        self.infomap = infomap
        self.max_depth = self.infomap.maxTreeDepth()

    def __getitem__(self, depth):
        """ Returns dictionary mapping low level modules to higher modules. """
        return self.group_modules(depth)

    def __call__(self, modules, level=None):
        """ Returns labels for modules cut to <level>. """
        return self.group(modules, level)

    @property
    def module_to_paths(self):
        return {m.moduleIndex(): m.path() for m in self.infomap.iterModules() if m.isLeafModule()}

    @property
    def node_to_leaf_module(self):
        return {n.physical_Id: n.moduleIndex() for n in self.infomap.iterLeafNodes()}

    @staticmethod
    def consolidate_values(adict):
        value_to_unique = {v:k for k,v in dict(enumerate(set(list(adict.values())))).items()}
        return {k: value_to_unique[v] for k,v in adict.items()}

    def group_modules(self, depth):
        module_to_cut_path = {m: self._cut_path(p, depth) for m, p in self.module_to_paths.items()}
        module_to_cut_module = self.consolidate_values(module_to_cut_path)
        return module_to_cut_module

    def _cut_path(self, path, depth):

        if len(path) <= depth:
            return path

        elif len(path)-1 == depth:
            return path[:-1]

        else:
            return self._cut_path(path[:-1], depth)

    def group(self, modules, level=0):

        if level is None:
            level = 0

        depth = self.max_depth - level - 1

        module_map = np.vectorize(self.group_modules(depth).get)

        return module_map(modules)


    # alternate more efficient method:
    # multilevel = imap.infomap.getMultilevelModules()
    # unique_paths = set([p[:depth] for p in multilevel.values()])
    # path_to_community = {path[:depth]: i for i, path  in dict(enumerate(unique_paths)).items()}
    # node_to_community = {node: path_to_community[path[:depth]] for node, path in multilevel.asdict().items()}

