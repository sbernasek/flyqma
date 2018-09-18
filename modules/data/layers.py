from os.path import join, isdir, exists
from os import listdir

from .images import ImageRGB

# create directory
        #dirpath = io.make_dir(self.path, force=True)


class Layer(ImageRGB):
    """
    Object represents a single RGB image layer.

    Attributes:
    path (str) - path to layer directory
    _id (int) - layer ID
    subdirs (dict) - {name: path} pairs for all subdirectories
    metadata (dict) - layer metadata
    labels (np.ndarray[int]) - segment ID mask
    cell_classifier (clones.annotation.classification.CellClassifier)
    graph (clones.annotation.graphs.Graph) - graph connecting cell centroids

    Inherited attributes:
    im (np.ndarray[float]) - 2D array of RGB pixel values
    channels (dict) - {color: channel_index} pairs
    shape (array like) - image dimensions
    mask (np.ndarray[bool]) - image mask
    labels (np.ndarray[int]) - segment ID mask
    """

    def __init__(self, im, path, cell_classifier=None):
        """
        Instantiate layer.

        Args:
        im (np.ndarray[float]) - 2D array of RGB pixel values
        path (str) - path to layer directory
        cell_classifier (clones.annotation.classification.CellClassifier)
        """

        # set layer ID
        layer_id = int(path.rsplit('/', maxsplit=1)[-1])
        self._id = layer_id

        # set path and subdirectories
        self.path = path
        self.find_subdirs()

        # load layer
        self.load()

        # set cell classifier
        self.cell_classifier = cell_classifier

        # call parent instantiation
        super().__init__(im, labels=self.labels)

    def initialize(self):
        """
        Initialize layer directory.

        Args:
        bits (int) - tif resolution
        """

        # make layers directory
        if not exists(self.path):
            mkdir(self.path)

        # make metadata file
        io = IO()
        metadata = dict(bg='', params=dict(segmentation_kw={}, graph_kw={}))
        io.write_json(join(self.path, 'metadata.json'), metadata)

        # make measurements file
        io.write_json(self.contours_path, {})

    def add_subdir(self, dirname, dirpath):
        """ Add subdirectory. """
        self.subdirs[dirname] = dirpath

    def find_subdirs(self):
        """ Find all subdirectories. """
        self.subdirs = {}
        for dirpath in listdir(self.path):
            if isdir(dirpath):
                dirname = dirpath.rsplit('/', maxsplit=1)[-1]
                self.add_subdir(dirname) = dirpath

    def load(self):
        """ Load layer. """

        # load metadata and extract background channel
        io = IO()
        self.metadata = io.read_json(join(self.path, 'metadata.json'))

        # load segment labels
        if 'segmentation' in self.subdirs.keys():
            self.load_labels()

        # load measurements
        self.load_measurements()

        # build graph
        self.build_graph(**self.metadata['params']['graph_kw'])

        # load selection
        if 'selection' in self.subdirs.keys():
            self.load_selection()

    def load_labels(self):
        """ Load segment labels. """
        segmentation_path = self.subdirs['segmentation']
        labels_path = join(segmentation_path, 'labels.npy')
        if exists(labels_path):
            self.labels = np.load(labels_path)
        else:
            self.labels = None

    def load_measurements(self):
        """ Load measurements. """
        df = pd.read_json(io.read_json(join(self.path, 'contours.json')))
        self.df = df

    def load_selection(self):
        """ Load selection. """

        # load selection metadata
        io = IO()
        selection_md = io.read_json(join(self.subdirs['selection'], 'md.json'))
        if selection_md is not None:
            self.include = bool(selection_md['include'])

        # load selection points
        pts = io.read_npy(join(self.subdirs['selection'], 'selection.npy'))
        self.selection_pts = pts

    def build_graph(self, **graph_kw):
        """
        Compile weighted graph connecting adjacent cells.

        Keyword Args:
        q (float) - edge length quantile above which edges are pruned
        weighted_by (str) - quantity used to weight edges
        """
        self.metadata['params']['graph_kw'] = graph_kw
        self.graph = WeightedGraph(self.df, **graph_kw)

    def annotate(self, cluster=False):
        """
        Assign genotype labels to cell measurements.

        Args:
        cluster (bool) - if True, add label for cell cluster
        """

        # assign single-cell classifier label
        self.df['genotype'] = self.cell_classifier(self.df)

        # assign cluster labels
        if cluster:
            annotation = Annotation(self.graph, self.cell_classifier)
            self.df['community_genotype'] = annotation(self.df)

    def mark_boundaries(self, basis='genotype', max_edges=0):
        """
        Mark clone boundaries by assigning a boundary label to all cells that share an edge with another cell from a different clone.

        Args:
        basis (str) - labels used to identify clones
        max_edges (int) - maximum number of edges for interior cells
        """

        # assign genotype to edges
        assign_genotype = np.vectorize(dict(self.df[basis]).get)
        edge_genotypes = assign_genotype(self.graph.edges)

        # find edges traversing clones
        boundaries = (edge_genotypes[:, 0] != edge_genotypes[:, 1])

        # get number of clone-traversing edges per node
        boundary_edges = self.graph.edges[boundaries]
        edge_counts = Counter(boundary_edges.ravel())

        # assign boundary label to nodes with too many clone-traversing edges
        boundary_nodes = [n for n, c in edge_counts.items() if c>max_edges]
        self.df['boundary'] = False
        self.df.loc[boundary_nodes, 'boundary'] = True

    def segment(self, bg='b', **kwargs):
        """
        Identify nuclear contours by running watershed segmentation on specified background channel.

        Args:
        bg (str) - background channel

        Keyword Arguments:
        preprocessing_kws (dict) - keyword arguments for image preprocessing
        seed_kws (dict) - keyword arguments for seed detection
        seg_kws (dict) - keyword arguments for segmentation
        min_area (int) - threshold for minimum segment size, px
        """
        self.metadata['bg'] = bg
        super().segment(bg=bg, **kwargs)

    def segment(self,
                bg='b',
                preprocessing_kws={},
                seed_kws={},
                seg_kws={},
                min_area=250):
        """
        Identify nuclear contours by running watershed segmentation on specified background channel.

        Args:
        bg (str) - background channel on which to segment RGB image
        preprocessing_kws (dict) - keyword arguments for image preprocessing
        seed_kws (dict) - keyword arguments for seed detection
        seg_kws (dict) - keyword arguments for segmentation
        min_area (int) - threshold for minimum segment size, px
        """

        # update metadata
        self.metadata['bg'] = bg
        segmentation_kw = dict(preprocessing_kws=preprocessing_kws,
                               seed_kws=seed_kws,
                               seg_kws=seg_kws,
                               min_area=min_area)
        self.metadata['params']['segmentation_kw'] = segmentation_kw

        # run segmentation on background
        background = self.get_channel(bg)
        background.preprocess(**preprocessing_kws)
        seg = Segmentation(background, seed_kws=seed_kws, seg_kws=seg_kws)
        seg.exclude_small_segments(min_area=min_area)

        # update labels
        self.labels = seg.labels

    def compile_measurements(self):
        """
        Compile measurements from segment labels.
        """

        # measure segment properties
        contours = self.measure()

        # construct dataframe
        columns = ['id',
                   'centroid_x', 'centroid_y',
                   'r', 'r_std',
                   'g', 'g_std',
                   'b', 'b_std',
                   'pixel_count']
        df = pd.DataFrame.from_records(contours, columns=columns)
        df['layer'] = self.layer_id

        # normalize by background
        for channel in 'rgb'.strip(self.metadata['bg']):
            df[channel+'_normalized'] = df[channel] / df[self.metadata['bg']]

        self.df = df

    def plot_graph(self,
                   channel='r',
                   figsize=(15, 15)
                   **image_kw,
                   **graph_kw):
        """
        Plot graph on top of relevant channel from RGB image.

        Args:
        channel (str) - RGB channel to visualize
        figsize (tuple) - figure size
        image_kw (dict) - keyword arguments for scalar image visualization
        graph_kw (dict) - keyword arguments for scalar image visualization

        Returns:
        fig (matplotlib.figures.Figure)
        """

        # create axis
        fig, ax = plt.subplots(figsize=figsize)

        # add image
        image = self.get_channel(channel)
        image.show(ax=ax, segments=False, **image_kw)

        # add graph
        self.graph.show(ax=ax, **graph_kw)

        return fig

    # def plot_annotation(self,
    #                     cmap='grey',
    #                     fig_kw={},
    #                     clone_kw={}):
    #     """
    #     Show annotation channel overlayed with clone segments.

    #     Args:
    #     cmap (matplotlib.colors.ColorMap)
    #     fig_kw (dict) - keyword arguments for layer visualization
    #     clone_kw (dict) - keyword arguments for clone visualization
    #     """

    #     if cmap == 'grey':
    #         cmap = plt.cm.Greys

    #     # get layer
    #     weighted_by = self.annotation.graph.weighted_by
    #     if 'normalized' in self.annotation.graph.weighted_by:
    #         im = self.get_channel(weighted_by.split('_')[0])
    #     else:
    #         im = self.get_channel(weighted_by)

    #     # show layer
    #     fig = im.show(segments=False, cmap=cmap, **fig_kw)

    #     # add clones
    #     self.annotation.plot_clones(fig.axes[0], **clone_kw)

    #     # mask background
    #     if self.annotation.fg_only:
    #         self.fg_mask.add_contourf(fig.axes[0], alpha=0.5)

    #     return fig

    def save_contours(self):
        """ Save measurements. """
        io = IO()
        io.write_json(join(self.path, 'contours.json'), self.df.to_json())

    def save_segmentation(self, **image_kw):
        """
        Save segmentation.

        image_kw: keyword arguments for segmentation image
        """
        dirpath = self.subdirs['segmentation']

        # save labels
        np.save(join(dirpath, 'labels.npy'), self.labels)

        # save segmentation image
        bg = self.get_channel(self.metadata['bg'], copy=False)
        fig = bg.show(segments=True)
        fig.axes[0].axis('off')
        fig.savefig(join(dirpath, 'segmentation.png'), **image_kw)
        fig.clf()
        plt.close(fig)
        gc.collect()

    def save_metadata(self):
        """ Save metadata. """
        io = IO()
        io.write_json(join(self.path, 'metadata.json'), self.metadata)

    def save(self,
             segmentation=True,
             annotation=False,
             dpi=100):
        """
        Save segmentation parameters and results.

        Args:
        segmentation (bool) - if True, save segmentation
        annotation (bool) - if True, save annotation
        dpi (int) - image resolution
        """

        # set image keyword arguments
        image_kw = dict(format='png',
                     dpi=dpi,
                     bbox_inches='tight',
                     pad_inches=0)

        # save contours
        self.save_contours()

        # save segmentation
        if segmentation:
            self.save_segmentation(**image_kw)

        # save metadata
        self.save_metadata()

        # # save annotation image
        # if annotation:
        #     im_path = os.path.join(dirpath, 'annotation.png')
        #     fig = self.plot_annotation()
        #     fig.savefig(im_path, **im_kw)
        #     fig.clf()
        #     plt.close(fig)
        #     gc.collect()

        return dirpath
