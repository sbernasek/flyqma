from os.path import join, isdir, exists
from os import listdir
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from collections import Counter

from .images import ImageRGB
from ..measure.segmentation import Segmentation
from ..spatial.graphs import WeightedGraph
from ..annotation.genotype import CommunityBasedGenotype
from ..annotation.labelers import CelltypeLabeler
from ..annotation.concurrency import ConcurrencyLabeler
from ..utilities.io import IO


class Layer(ImageRGB):
    """
    Object represents a single RGB image layer.

    Attributes:
    df (pd.DataFrame) - cell measurement data
    path (str) - path to layer directory
    _id (int) - layer ID
    subdirs (dict) - {name: path} pairs for all subdirectories
    metadata (dict) - layer metadata
    labels (np.ndarray[int]) - segment ID mask
    classifier (CellClassifier) - callable that assigns genotypes to cells
    graph (Graph) - graph connecting cell centroids
    include (bool) - if True, layer was manually marked for inclusion

    Inherited attributes:
    im (np.ndarray[float]) - 2D array of RGB pixel values
    channels (dict) - {color: channel_index} pairs
    shape (array like) - image dimensions
    mask (np.ndarray[bool]) - image mask
    labels (np.ndarray[int]) - segment ID mask
    """

    def __init__(self, path, im=None, classifier=None, load_all=True):
        """
        Instantiate layer.

        Args:
        path (str) - path to layer directory
        im (np.ndarray[float]) - 2D array of RGB pixel values
        classifier (CellClassifier) - callable that assigns genotypes to cells
        load_all (bool) - if True, load labels and build graph
        """

        # set layer ID
        layer_id = int(path.rsplit('/', maxsplit=1)[-1])
        self._id = layer_id

        # set path and subdirectories
        self.path = path
        self.find_subdirs()

        # load layer
        self.load(load_all=load_all)

        # set cell classifier and annotate measurements
        if classifier is not None:
            self.classifier = classifier
            self.annotate()
            self.build_graph(**self.metadata['params']['graph_kw'])
            self.mark_boundaries(basis='genotype', max_edges=1)
            self.assign_concurrency()

        # load labels and instantiate RGB image if image was provided
        if im is not None:
            self.load_labels()
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

    def make_subdir(self, dirname):
        """ Make subdirectory. """
        dirpath = join(self.path, dirname)
        if not exists(dirpath):
            mkdir(dirpath)
        self.add_subdir(dirname, dirpath)

    def add_subdir(self, dirname, dirpath):
        """ Add subdirectory. """
        self.subdirs[dirname] = dirpath

    def find_subdirs(self):
        """ Find all subdirectories. """
        self.subdirs = {}
        for dirname in listdir(self.path):
            dirpath = join(self.path, dirname)
            if isdir(dirpath):
                self.add_subdir(dirname, dirpath)

    def load(self, load_all=True):
        """
        Load layer.

        Args:
        load_all (bool) - if True, load labels and build graph
        """

        # load metadata and extract background channel
        self.load_metadata()

        # load measurements
        if 'segmentation' in self.subdirs.keys():
            self.load_measurements()

        # load and apply selection
        if 'selection' in self.subdirs.keys():
            self.load_selection()
            self.apply_selection()

        # load and apply correction
        if 'correction' in self.subdirs.keys():
            self.apply_correction()


    def load_metadata(self):
        """ Load metadata. """
        path = join(self.path, 'metadata.json')
        if exists(path):
            io = IO()
            self.metadata = io.read_json(path)

    def load_labels(self):
        """ Load segment labels if they are available. """
        labels = None
        if 'segmentation' in self.subdirs.keys():
            segmentation_path = self.subdirs['segmentation']
            labels_path = join(segmentation_path, 'labels.npy')
            if exists(labels_path):
                labels = np.load(labels_path)
        self.labels = labels

    def load_measurements(self):
        """ Load raw measurements. """
        io = IO()
        path = self.subdirs['segmentation']
        self.df = pd.read_json(io.read_json(join(path, 'measurements.json')))

    def load_selection(self):
        """ Load selection. """
        io = IO()
        selection_md = io.read_json(join(self.subdirs['selection'], 'md.json'))
        if selection_md is not None:
            self.include = bool(selection_md['include'])

    def apply_selection(self):
        """
        Adds a "selected" attribute to the measurements dataframe. The attribute is true for cells that fall within the selection boundary.
        """

        # load selection boundary
        io = IO()
        bounds = io.read_npy(join(self.subdirs['selection'], 'selection.npy'))

        # add selected attribute to cell measurement data
        self.df['selected'] = False

        if self.include:

            # construct matplotlib path object
            path = Path(bounds, closed=False)

            # mark cells as within or outside the selection boundary
            cell_positions = self.df[['centroid_x', 'centroid_y']].values
            self.df['selected'] = path.contains_points(cell_positions)

    def load_correction(self):
        """
        Load linear background correction.

        Returns:
        correction (LayerCorrection)
        """
        return LayerCorrection.load(self)

    def apply_correction(self):
        """
        Adds a "selected" attribute to the measurements dataframe. The attribute is true for cells that fall within the selection boundary.
        """

        # load correction coefficients and X/Y variables
        io = IO()
        data = io.read_json(join(self.subdirs['correction'], 'data.json'))

        # get independent/dependent variables
        xvar = data['params']['xvar']
        yvar = data['params']['yvar']

        # get linear model coefficients
        b, m = data['coefficients']

        # apply correction
        trend = b+m*self.df[xvar].values
        self.df[yvar+'p'] = trend
        self.df[yvar+'_corrected'] = self.df[yvar] - trend

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
        Assign genotype and celltype labels to cell measurements.

        Args:
        cluster (bool) - if True, add community and community genotype labels
        """

        # assign single-cell classifier label
        self.df['genotype'] = self.classifier(self.df)

        # assign cluster labels
        if cluster:
            assign_genotypes = CommunityBasedGenotype.from_layer(self)
            assign_genotypes(self.df)

        # assign celltype labels
        celltype_labels = {0:'m', 1:'h', 2:'w', -1:'none'}
        assign_celltypes = CelltypeLabeler(labels=celltype_labels)
        assign_celltypes(self.df)

    def assign_concurrency(self, min_pop=5, max_distance=10):
        """
        Add boolean 'concurrent_<cell type>' field to cell measurement data for each unique cell type.

        Args:
        min_pop (int) - minimum population size for inclusion of cell type
        max_distance (float) - maximum distance threshold for inclusion
        """
        assign_concurrency = ConcurrencyLabeler(min_pop=min_pop,
                                                max_distance=max_distance)
        assign_concurrency(self.df)

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

        # extract and preprocess background
        background = self.get_channel(bg)
        background.preprocess(**preprocessing_kws)

        # run segmentation
        seg = Segmentation(background, seed_kws=seed_kws, seg_kws=seg_kws)
        seg.exclude_small_segments(min_area=min_area)

        # update segment labels
        self.labels = seg.labels

    def measure(self):
        """
        Measure properties of cell segments to generate cell measurement data.
        """

        # measure segment properties
        measurements = super().measure()

        # construct dataframe
        columns = ['id',
                   'centroid_x', 'centroid_y',
                   'r', 'r_std',
                   'g', 'g_std',
                   'b', 'b_std',
                   'pixel_count']
        df = pd.DataFrame.from_records(measurements, columns=columns)
        df['layer'] = self.layer_id

        # normalize by background intensity
        for channel in 'rgb'.strip(self.metadata['bg']):
            df[channel+'_normalized'] = df[channel] / df[self.metadata['bg']]


        # add segmentation directory
        self.make_subdir('segmentation')
        dirpath = self.subdirs['segmentation']

        # save raw measurements
        io = IO()
        io.write_json(join(dirpath, 'measurements.json'), df.to_json())

    def plot_graph(self,
                   channel='r',
                   figsize=(15, 15),
                   image_kw={},
                   graph_kw={}):
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

        # add segmentation directory
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
