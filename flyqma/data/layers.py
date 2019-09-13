from os.path import join, isdir, exists
from os import listdir, mkdir
from shutil import rmtree
import gc
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from collections import Counter

from ..visualization import *
from ..utilities import IO

# import measurement objects
from ..measurement import Segmentation
from ..measurement import Measurements

# import annotation objects
from ..annotation import WeightedGraph
from ..annotation import Annotation
from ..annotation import ConcurrencyLabeler
from ..annotation import CloneBoundaries
from ..annotation import CelltypeLabeler

# import bleedthrough correction objects
from ..bleedthrough import LayerCorrection

# import image base class
from .images import ImageMultichromatic
from .silhouette import SilhouetteLayerIO

# import default parameters
from .defaults import Defaults
defaults = Defaults()


class LayerVisualization:
    """ Methods for visualizing a layer. """

    @default_figure
    def plot_graph(self, channel,
                   figsize=(15, 15),
                   image_kw={},
                   graph_kw={},
                   ax=None):
        """
        Plot graph on top of relevant image channel.

        Args:

            channel (str) - color channel to visualize

            figsize (tuple) - figure size

            image_kw (dict) - keyword arguments for scalar image visualization

            graph_kw (dict) - keyword arguments for scalar image visualization

        """

        # add image
        if channel is not None:
            image = self.get_channel(channel)
            image.show(ax=ax, segments=False, **image_kw)

        # add graph
        self.graph.show(ax=ax, **graph_kw)

    def plot_boundary(self, ax,
                        label,
                        label_by='genotype',
                        color='r',
                        alpha=70,
                        **kwargs):
        """ Plot boundary of <label_by> groups with <label> on <ax>. """

        # add labels to ephemeral copy of graph data
        graph = self.graph.copy()
        graph.data[label_by] = self.data[label_by]

        # plot clone boundaries
        bounds = CloneBoundaries(graph, label_by=label_by, alpha=alpha)
        bounds.plot_boundary(label, color=color, ax=ax, **kwargs)

    def plot_boundaries(self, ax,
                        label_by='genotype',
                        cmap=plt.cm.bwr,
                        alpha=70,
                        **kwargs):
        """ Plot boundaries of all <label_by> groups on <ax>. """

        # add labels to ephemeral copy of graph data
        graph = self.graph.copy()
        graph.data[label_by] = self.data[label_by]

        # plot clone boundaries
        bounds = CloneBoundaries(graph, label_by=label_by, alpha=alpha)
        bounds.plot_boundaries(cmap=cmap, ax=ax, **kwargs)

    def _build_mask(self, values,
                   interior_only=False,
                   selection_only=False,
                   null_value=-1):
        """
        Use <values> to construct an image mask.

        Args:

            values (array like) - value/label for each segment

            interior_only (bool) - if True, excludes clone borders

            selection_only (bool) - if True, only include selected region

            null_value (int) - value used to fill unused pixels

        Returns:

            mask (np.ma.Maskedarray) - masked image in which foreground segments are replaced with the specified values

        """

        # build dictionary mapping segments to values
        segment_to_value = dict(zip(self.data.segment_id, values))
        segment_to_value[0] = null_value

        # exclude borders
        if interior_only:
            boundary = self.data[self.data.boundary]
            boundary_to_black = {x: null_value for x in boundary.segment_id}
            segment_to_value.update(boundary_to_black)

        # exclude cells not included in selection
        if selection_only:
            excluded = self.data[~self.data.selected]
            excluded_to_black = {x: null_value for x in excluded.id}
            segment_to_value.update(excluded_to_black)

        # construct mask
        segment_to_value = np.vectorize(segment_to_value.get)
        mask = segment_to_value(self.labels)
        mask = np.ma.MaskedArray(mask, mask==null_value)

        return mask

    def build_attribute_mask(self, attribute,
                             interior_only=False,
                             selection_only=False,
                             **kwargs):
        """
        Use <attribute> value for each segment to construct an image mask.

        Args:

            attribute (str) - attribute used to label each segment

            interior_only (bool) - if True, excludes clone borders

            selection_only (bool) - if True, only include selected region

        Returns:

            mask (np.ma.Maskedarray) - masked image in which foreground segments are replaced with the attribute values

        """

        return self._build_mask(self.data[attribute].values,
                                interior_only=interior_only,
                                selection_only=selection_only,
                                **kwargs)

    def build_classifier_mask(self, classifier,
                   interior_only=False,
                   selection_only=False,
                   **kwargs):
        """
        Use segment <classifier> to construct an image mask.

        Args:

            classifier (annotation.Classifier object)

            interior_only (bool) - if True, excludes clone borders

            selection_only (bool) - if True, only include selected region

        Returns:

            mask (np.ma.Maskedarray) - masked image in which foreground segments are replaced with the assigned labels

        """
        return self._build_mask(classifier(self.data),
                                interior_only=interior_only,
                                selection_only=selection_only,
                                **kwargs)


class LayerIO(SilhouetteLayerIO):
    """
    Methods for saving and loading Layer objects and their subcomponents.
    """

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

    def save_metadata(self):
        """ Save metadata. """
        io = IO()
        io.write_json(join(self.path, 'metadata.json'), self.metadata)

    def save_segmentation(self, image, **kwargs):
        """
        Save segment labels, and optionally save a segmentation image.

        Args:

            image (bool) - if True, save segmentation image

            kwargs: keyword arguments for image rendering

        """
        dirpath = self.subdirs['segmentation']

        # save segment labels
        np.save(join(dirpath, 'labels.npy'), self.labels)

        # save segmentation image
        if image:
            bg = self.get_channel(self.metadata['bg'], copy=False)
            fig = bg.show(segments=True)
            fig.axes[0].axis('off')
            fig.savefig(join(dirpath, 'segmentation.png'), **kwargs)
            fig.clf()
            plt.close(fig)
            gc.collect()

    def save_measurements(self):
        """ Save raw measurements. """

        # get segmentation directory
        path = join(self.subdirs['measurements'], 'measurements.hdf')

        # save raw measurements
        self.measurements.to_hdf(path, 'measurements', mode='w')

    def save_processed_data(self):
        """ Save processed measurement data. """

        path = join(self.subdirs['measurements'], 'processed.hdf')
        self.data.to_hdf(path, 'data', mode='w')

    def save_annotator(self, image=False, **kwargs):
        """
        Save annotator instance.

        Args:

            image (bool) - if True, save annotation images

            kwargs: keyword arguments for image rendering

        """
        path = self.subdirs['annotation']
        self.annotator.save(path, image=image, **kwargs)

    def save(self,
             segmentation=True,
             measurements=True,
             processed_data=True,
             annotator=False,
             segmentation_image=False,
             annotation_image=False):
        """
        Save segmentation parameters and results.

        Args:

            segmentation (bool) - if True, save segmentation

            measurements (bool) - if True, save measurement data

            processed_data (bool) - if True, save processed measurement data

            annotator (bool) - if True, save annotator

            segmentation_image (bool) - if True, save segmentation image

            annotation_image (bool) - if True, save annotation image

        """

        # set image keyword arguments
        image_kw = dict(format='png',
                     dpi=100,
                     bbox_inches='tight',
                     pad_inches=0,
                     transparent=True,
                     rasterized=True)

        # save segmentation
        if segmentation:
            self.make_subdir('segmentation')
            self.save_segmentation(image=segmentation_image, **image_kw)

        # save measurements
        if measurements:
            self.make_subdir('measurements')
            self.save_measurements()

        # save processed data
        if processed_data and self.data is not None:
            self.data = self.process_measurements(self.measurements)
            self.save_processed_data()

        # save annotation
        if annotator and self.annotator is not None:
            self.make_subdir('annotation')
            self.save_annotator(image=annotation_image, **image_kw)

        # save metadata
        self.save_metadata()

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
        path = join(self.subdirs['measurements'], 'measurements.hdf')
        self.measurements = pd.read_hdf(path, 'measurements')

    def load_processed_data(self):
        """ Load processed data from file. """
        path = join(self.subdirs['measurements'], 'processed.hdf')
        self.data = pd.read_hdf(path, 'data')

    def load_annotator(self):
        """ Load annotator instance. """
        self.annotator = Annotation.load(self.subdirs['annotation'])

    def load_inclusion(self):
        """ Load inclusion flag. """
        io = IO()
        selection_md = io.read_json(join(self.subdirs['selection'], 'md.json'))
        if selection_md is not None:
            self.include = bool(selection_md['include'])

    def load_correction(self):
        """
        Load linear background correction.

        Returns:

           correction (LayerCorrection)

        """
        return LayerCorrection.load(self)

    def load(self, process=False, graph=True):
        """
        Load layer.

        Args:

            process (bool) - if True, re-process the measurement data

            graph (bool) - if True, load weighted graph

        """

        # load metadata and extract background channel
        self.load_metadata()

        # load inclusion data
        if 'selection' in self.subdirs.keys():
            self.load_inclusion()

        # if layer is not included, skip it
        if not self.include:
            return None

        # check whether annotation exists
        if 'annotation' in self.subdirs.keys() and process:

            if self.annotator is not None:
                raise UserWarning('Layer was instantiated with a stack-level annotation instance, but a second annotation instance was found within the layer directory. Resolve this conflict before continuing.')

            # load annotator
            self.load_annotator()

        # check whether segmentation exists and load raw measurement data
        if 'measurements' in self.subdirs.keys():
            self.load_measurements()

        # if processing measurements, ensure that graph is built
        if process:
            graph = True

        # build graph
        if graph and 'graph_weighted_by' in self.metadata['params'].keys():
            graph_weighted_by = self.metadata['params']['graph_weighted_by']
            graph_kw = self.metadata['params']['graph_kw']
            self.build_graph(graph_weighted_by, **graph_kw)

        # check whether measurements are available
        if 'measurements' in self.subdirs.keys():
            path = join(self.subdirs['measurements'], 'processed.hdf')

            # load processed data
            if not process and exists(path):
                self.load_processed_data()

            # otherwise, process raw measurement data
            else:
                self.data = self.process_measurements(self.measurements)


class Layer(LayerIO, ImageMultichromatic, LayerVisualization):
    """
    Object represents a single imaged layer.

    Attributes:

        measurements (pd.DataFrame) - raw cell measurement data

        data (pd.DataFrame) - processed cell measurement data

        path (str) - path to layer directory

        _id (int) - layer ID

        subdirs (dict) - {name: path} pairs for all subdirectories

        metadata (dict) - layer metadata

        labels (np.ndarray[int]) - segment ID mask

        annotator (Annotation) - object that assigns labels to measurements

        graph (Graph) - graph connecting cell centroids

        include (bool) - if True, layer was manually marked for inclusion

    Inherited attributes:

        im (np.ndarray[float]) - 3D array of pixel values

        shape (array like) - image dimensions

        mask (np.ndarray[bool]) - image mask

        labels (np.ndarray[int]) - segment ID mask

    Properties:

        colordepth (int) - number of color channels

    """

    def __init__(self, path, im=None, annotator=None):
        """
        Instantiate layer.

        Args:

            path (str) - path to layer directory

            im (np.ndarray[float]) - 3D array of pixel values

            annotator (Annotation) - object that assigns labels to measurements

        """

        # set layer ID
        layer_id = int(path.rsplit('/', maxsplit=1)[-1])
        self._id = layer_id
        self.xykey = ['centroid_x', 'centroid_y']

        # set path and subdirectories
        self.path = path

        # make layers directory
        if not exists(self.path):
            self.initialize()
        self.find_subdirs()

        # load inclusion; defaults to True
        if 'selection' in self.subdirs.keys():
            if len(listdir(self.subdirs['selection'])) == 0:
                self.include = True
            else:
                self.load_inclusion()
        else:
            self.include = True

        # set annotator
        self.annotator = annotator

        # load labels and instantiate image
        if im is not None:
            self.load_labels()
            super().__init__(im, labels=self.labels)

    @property
    def colordepth(self):
        """ Number of color channels. """
        return self.im.shape[-1]

    @property
    def bg_key(self):
        """ DataFrame key for background channel. """
        return self._to_key(self.metadata['bg'])

    def initialize(self):
        """

        Initialize layer directory by:

            - Creating a layer directory
            - Removing existing segmentation directory
            - Saving metadata to file

        """

        # make layers directory
        if not exists(self.path):
            mkdir(self.path)
        self.subdirs = {}

        # remove existing segmentation/annotation/measurement directories
        for key in ('segmentation', 'measurements', 'annotation'):
            if key in self.subdirs.keys():
                rmtree(self.subdirs[key])

        # make metadata file
        segmentation_kw = dict(preprocessing_kws={}, seed_kws={}, seg_kws={})
        params = dict(segmentation_kw=segmentation_kw, graph_kw={})
        metadata = dict(bg=None, params=params)

        # save metadata
        IO().write_json(join(self.path, 'metadata.json'), metadata)

    def process_measurements(self, measurements):
        """
        Augment measurements by:
            1. incorporating manual selection boundary
            2. correcting for fluorescence bleedthrough
            3. assigning measurement labels
            4. marking clone boundaries
            5. assigning label concurrency information

        Operations 3-5 require construction of a WeightedGraph object.

        Args:

            measurements (pd.DataFrame) - raw measurement data

        Returns:

            data (pd.DataFrame) - processed measurement data

        """

        # copy raw measurements
        data = deepcopy(self.measurements)

        # load and apply selection
        if 'selection' in self.subdirs.keys():
            self.apply_selection(data)

        # load and apply correction
        if 'correction' in self.subdirs.keys():
            self.apply_correction(data)

        # annotate measurements (opt to load labels rather than annotate again)
        self.annotate(data)

        return data

    def annotate(self, data=None):
        """ Apply annotation to <data>. """

        if data is None:
            data = self.data

        if self.annotator is not None and self.graph is not None:
            self._apply_annotation(data)
            self.mark_boundaries(data, basis='genotype', max_edges=1)
            self.apply_concurrency(data, basis='genotype')

    def apply_normalization(self, data):
        """
        Normalize fluorescence intensity measurements by measured background channel intensity.

        Args:

            data (pd.DataFrame) - processed cell measurement data

        """

        # get background channel from metadata
        bg = self.metadata['bg']

        # apply normalization to each foreground channel
        for fg in range(self.colordepth):
            if fg == bg:
                continue
            fg_key = self._to_key(fg)
            data['{:s}_normalized'.format(fg_key)] = data[fg_key]/data[self.bg_key]

    def apply_selection(self, data):
        """
        Adds a "selected" attribute to the measurements dataframe. The attribute is true for cells that fall within the selection boundary.

        Args:

            data (pd.DataFrame) - processed cell measurement data

        """

        # load selection boundary
        io = IO()
        bounds = io.read_npy(join(self.subdirs['selection'], 'selection.npy'))

        # add selected attribute to cell measurement data
        data['selected'] = False

        if self.include:

            # construct matplotlib path object
            path = Path(bounds, closed=False)

            # mark cells as within or outside the selection boundary
            cell_positions = data[self.xykey].values
            data['selected'] = path.contains_points(cell_positions)

    def apply_correction(self, data):
        """
        Adds bleedthrough-corrected fluorescence levels to the measurements dataframe.

        Args:

            data (pd.DataFrame) - processed cell measurement data

        """

        # load correction coefficients and X/Y variables
        io = IO()
        cdata = io.read_json(join(self.subdirs['correction'], 'data.json'))

        # get independent/dependent variables
        xvar = cdata['params']['xvar']
        yvar = cdata['params']['yvar']
        bgvar = self.metadata['bg']
        if type(xvar) == int:
            xvar = 'ch{:d}'.format(xvar)
        if type(yvar) == int:
            yvar = 'ch{:d}'.format(yvar)
        if type(bgvar) == int:
            bgvar = 'ch{:d}'.format(bgvar)

        # get linear model coefficients
        b, m = cdata['coefficients']

        # apply correction
        trend = b + m * data[xvar].values
        data[yvar+'_predicted'] = trend
        data[yvar+'c'] = data[yvar] - trend
        data[yvar+'c_normalized'] = data[yvar+'c'] / data[bgvar]

    def build_graph(self, weighted_by, **graph_kw):
        """
        Compile weighted graph connecting adjacent cells.

        Args:

            weighted_by (str) - attribute used to weight edges

            graph_kw: keyword arguments, including:

                xykey (list) - attribute keys for node x/y positions

                logratio (bool) - if True, weight edges by log ratio

                distance (bool) - if True, weights edges by distance

        """

        # store metadata for graph reconstruction
        self.metadata['params']['graph_weighted_by'] = weighted_by
        self.metadata['params']['graph_kw'] = graph_kw

        # build graph
        self.graph = WeightedGraph(self.measurements, weighted_by, **graph_kw)

    def train_annotator(self, attribute,
                        save=False,
                        logratio=True,
                        **kwargs):
        """
        Train an Annotation model on the measurements in this layer.

        Args:

            attribute (str) - measured attribute used to determine labels

            save (bool) - if True, save model selection routine

            logratio (bool) - if True, weight edges by relative attribute value

            kwargs: keyword arguments for Annotation, including:

                sampler_type (str) - either 'radial', 'neighbors', 'community'

                sampler_kwargs (dict) - keyword arguments for sampler

                min_num_components (int) - minimum number of mixture components

                max_num_components (int) - maximum number of mixture components

                addtl_kwargs: keyword arguments for Classifier

        Returns:

            selector (ModelSelection object)

        """

        # instantiate annotator
        self.annotator = Annotation(attribute, **kwargs)

        # build graph and use it to train annotator
        self.build_graph(attribute, logratio=logratio)
        selector = self.annotator.train(self.graph)

        # save trained annotator
        if save:
            self.save_metadata()
            self.make_subdir('annotation')
            selector.save(self.subdirs['annotation'])

        return selector

    def _apply_annotation(self, data,
                          label='genotype',
                          **kwargs):
        """
        Assign labels to cell measurements.

        Args:

            data (pd.DataFrame) - processed cell measurement data

            label (str) - attribute name for predicted genotype

            kwargs: keyword arguments for Annotator.annotate()

        """
        if self.annotator is not None and self.graph is not None:
            data[label] = self.annotator(self.graph, **kwargs)

    def apply_annotation(self, label='genotype', **kwargs):
        """
        Assign labels to cell measurements in place.

        Args:

            label (str) - attribute name for predicted genotype

            kwargs: keyword arguments for Annotator.annotate()

        """
        self._apply_annotation(self.data, label=label, **kwargs)

    def apply_concurrency(self, data, basis='genotype',
                          min_pop=5, max_distance=10):
        """
        Add boolean 'concurrent_<cell type>' field to cell measurement data for each unique cell type.

        Args:

            data (pd.DataFrame) - processed cell measurement data

            basis (str) - attribute on which concurrency is established

            min_pop (int) - minimum population size for inclusion of cell type

            max_distance (float) - maximum distance threshold for inclusion

        """
        assign_concurrency = ConcurrencyLabeler(attribute=basis,
                                                label_values=(0,1,2),
                                                min_pop=min_pop,
                                                max_distance=max_distance)
        assign_concurrency(data)

    def mark_boundaries(self, data, basis='genotype', max_edges=0):
        """
        Mark clone boundaries by assigning a boundary label to all cells that share an edge with another cell from a different clone.

        Args:

            data (pd.DataFrame) - processed cell measurement data

            basis (str) - labels used to identify clones

            max_edges (int) - maximum number of edges for interior cells

        """

        # assign genotype to edges
        assign_genotype = np.vectorize(dict(data[basis]).get)
        edge_genotypes = assign_genotype(self.graph.edges)

        # find edges traversing clones
        boundaries = (edge_genotypes[:, 0] != edge_genotypes[:, 1])

        # get number of clone-traversing edges per node
        boundary_edges = self.graph.edges[boundaries]
        edge_counts = Counter(boundary_edges.ravel())

        # assign boundary label to nodes with too many clone-traversing edges
        boundary_nodes = [n for n, c in edge_counts.items() if c>max_edges]
        data['boundary'] = False
        data.loc[boundary_nodes, 'boundary'] = True

    def segment(self, channel,
                preprocessing_kws={},
                seed_kws={},
                seg_kws={},
                min_area=250):
        """
        Identify nuclear contours by running watershed segmentation on specified background channel.

        Args:

            channel (int) - channel index on which to segment image

            preprocessing_kws (dict) - keyword arguments for image preprocessing

            seed_kws (dict) - keyword arguments for seed detection

            seg_kws (dict) - keyword arguments for segmentation

            min_area (int) - threshold for minimum segment size, px

        Returns:

            background (ImageScalar) - background image post-processing

        """

        # append default parameter values
        preprocessing_kws = defaults('preprocessing', preprocessing_kws)
        seed_kws = defaults('seeds', seed_kws)
        seg_kws = defaults('segmentation', seg_kws)

        # store parameters in metadata
        self.metadata['bg'] = channel
        segmentation_kw = dict(preprocessing_kws=preprocessing_kws,
                               seed_kws=seed_kws,
                               seg_kws=seg_kws,
                               min_area=min_area)
        self.metadata['params']['segmentation_kw'] = segmentation_kw

        # extract and preprocess background
        background = self.get_channel(channel)
        background.preprocess(**preprocessing_kws)

        # run segmentation
        seg = Segmentation(background, seed_kws=seed_kws, seg_kws=seg_kws)

        # exclude small segments
        seg.exclude_small_segments(min_area=min_area)

        # update segment labels
        self.labels = seg.labels
        background.labels = seg.labels

        # update cell measurements
        self.measure()

        return background

    def measure(self):
        """
        Measure properties of cell segments. Raw measurements are stored under in the 'measurements' attribute, while processed measurements are stored in the 'data' attribute.
        """

        # measure segment properties
        measurements = Measurements(self.im, self.labels)
        measurements = measurements.build_dataframe()

        # assign layer id, apply normalization, and save measurements
        measurements['layer'] = self._id
        self.apply_normalization(measurements)
        self.measurements = measurements

        # process raw measurement data
        self.data = self.process_measurements(self.measurements)
