from os.path import join, isdir, exists
from os import listdir, mkdir
from shutil import rmtree
import gc
from copy import deepcopy
import pandas as pd
import numpy as np
from scipy.ndimage import binary_erosion
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
from .silhouette_write import WriteSilhouetteLayer

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

            channel (str) - fluorescence channel to visualize

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
            msg = 'Boundary attribute not found. Annotate and try again.'
            assert 'boundary' in self.data.keys(), msg
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


class LayerIO(WriteSilhouetteLayer):
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
        assert self.has_image, 'Image unavailable. Load image and try again.'
        return LayerCorrection.load(self)

    def load(self, use_cache=True, graph=True):
        """
        Load layer.

        Args:

            use_cache (bool) - if True, use cached measurement data, otherwise re-process the measurement data

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
        if 'annotation' in self.subdirs.keys() and not use_cache:

            if self.annotator is not None:
                raise UserWarning('Layer was instantiated with a stack-level annotation instance, but a second annotation instance was found within the layer directory. Resolve this conflict before continuing.')

            # load annotator
            self.load_annotator()

        # check whether segmentation exists and load raw measurement data
        if 'measurements' in self.subdirs.keys():
            self.load_measurements()

        # if processing measurements, ensure that graph is built
        if not use_cache:
            graph = True

        # build graph
        if graph and 'graph_weighted_by' in self.metadata['params'].keys():
            graph_weighted_by = self.metadata['params']['graph_weighted_by']
            graph_kw = self.metadata['params']['graph_kw']
            self.build_graph(graph_weighted_by, **graph_kw)
        else:
            self.graph = None

        # check whether cached measurements are available
        if 'measurements' in self.subdirs.keys():
            path = join(self.subdirs['measurements'], 'processed.hdf')

            # load processed data
            if use_cache and exists(path):
                self.load_processed_data()

            # otherwise, process raw measurement data
            else:
                self.data = self.process_measurements(self.measurements)


class LayerProperties:
    """
    Properties for Layer class:

        color_depth (int) - number of fluorescence channels

        num_cells (int) - number of cells detected by segmentation

        bg_key (str) - key for channel used to generate segmentation

        has_image (bool) - if True, image is loaded into memory

        is_segmented (bool) - if True, layer has been segmented

        has_trained_annotator (bool) - if True, layer has a trained annotator

    """

    @property
    def color_depth(self):
        """ Number of color channels. """
        return self.im.shape[-1]

    @property
    def num_cells(self):
        """ Number of cells detected by segmentation. """
        return len(self.data) if self.data is not None else None

    @property
    def bg_key(self):
        """ DataFrame key for background channel. """
        return self._to_key(self.metadata['bg'])

    @property
    def has_image(self):
        """ True if image is available. """
        return self.im is not None

    @property
    def is_segmented(self):
        """ True if measurement data are available. """
        return self.measurements is not None

    @property
    def has_trained_annotator(self):
        """ Returns True if trained annotator is available. """
        return self.annotator is not None


class LayerMeasurement:
    """

    Measurement related methods for Layer class.

    """

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

            background (ImageScalar) - background image (after processing)

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
                               min_area=min_area,
                               imported=False)
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
        self.data = self.process_measurements(measurements)

    def apply_normalization(self, data):
        """
        Normalize fluorescence intensity measurements by measured background channel intensity.

        Args:

            data (pd.DataFrame) - processed cell measurement data

        """

        # get background channel from metadata
        bg = self.metadata['bg']

        # apply normalization to each foreground channel
        for fg in range(self.color_depth):
            if fg == bg:
                continue
            fg_key = self._to_key(fg)
            data['{:s}_normalized'.format(fg_key)] = data[fg_key]/data[self.bg_key]

    def import_segmentation_mask(self, path, channel,
                                save=True,
                                save_image=True):
            """
            Import external segmentation mask and use it to generate measurements.

            Provided mask must contain a 2-D array of positive integers in which a values of zero denotes the image background.

            Args:

                path (str) - path to segmentation mask

                channel (int) - fluorescence channel used for segmentation

                save (bool) - if True, copy segmentation to stack directory

                save_image (bool) - if True, save segmentation image

            """

            assert exists(path), 'File does not exist.'

            io = IO()
            mask = io.read_npy(path)

            int_types = (int, np.int32, np.int64)
            assert mask.dtype in int_types, 'Mask does not contain integers.'
            assert mask.shape == self.shape, 'Mask dimensions are incorrect.'
            assert mask.min() >= 0, 'Mask contains values less than zero.'

            # set segmentation mask and generate measurements
            self.labels = mask
            self.metadata['bg'] = channel
            self.measure()

            # optionally copy mask to stack directory
            if save:
                self.metadata['params']['segmentation_kw']=dict(imported=True)
                self.save_metadata()

                self.make_subdir('segmentation')
                self.save_segmentation(save_image)

                self.make_subdir('measurements')
                self.save_measurements()


class LayerROI:
    """

    ROI related methods for Layer class.

    """

    @staticmethod
    def _apply_roi_vertices(data, xykey, roi_vertices):
        """
        Label cells within a specified region of interest.

        Args:

            data (pd.DataFrame) - cell measurement data

            roi_vertices (np.ndarray[int], N x 2) - vertices bounding ROI

        """

        # add selected attribute to cell measurement data
        data['selected'] = False

        # construct matplotlib path object
        path = Path(roi_vertices, closed=False)

        # mark cells as within or outside the selection boundary
        xy_positions = data[xykey].values
        data['selected'] = path.contains_points(xy_positions)

    @staticmethod
    def sort_clockwise(xycoords):
        """ Returns clockwise-sorted xy coordinates. """
        return xycoords[:, np.argsort(np.arctan2(*(xycoords.T - xycoords.mean(axis=1)).T))]

    @classmethod
    def mask_to_vertices(cls, mask):
        """
        Convert boolean mask to a list of vertices defining the border around the largest contiguous region.

        Args:

            mask (np.ndarray[bool]) - ROI mask, where True denotes the region. Note that the mask may only contain one contiguous component.

        Returns:

            vertices (np.ndarray[int]) - N x 2 array of vertices

        """

        borders = (mask != binary_erosion(mask, structure=np.ones((3, 3))))
        vertices = cls.sort_clockwise(np.asarray(borders.nonzero()))
        return vertices.T

    def import_roi_mask(self, path, save=True):
        """
        Import external ROI mask and use it to label measurement data.

        Provided mask must contain a 2-D boolean array with the same dimensions as the raw image. True values denote the ROI. The mask may only contain a single contiguous ROI.

        Args:

            path (str) - path to ROI mask

            save (bool) - if True, copy ROI mask to stack directory

        """

        assert exists(path), 'File does not exist.'

        # read mask and make sure it's valid
        io = IO()
        mask = io.read_npy(path)
        assert mask.min()>=0 and mask.max()<=1, 'Mask is not boolean.'
        assert mask.shape == self.shape, 'Mask dimensions are incorrect.'
        mask = mask.astype(bool)

        # convert mask to vertices and apply to measurement data
        vertices = self.mask_to_vertices(mask)
        self._apply_roi_vertices(self.data, self.xykey, vertices)

        # save ROI mask to stack directory
        if save:
            self.make_subdir('selection')
            selection_path = self.subdirs['selection']
            io = IO()
            io.write_npy(join(selection_path, 'selection.npy'), vertices)
            md = dict(include=True)
            io.write_json(join(selection_path, 'md.json'), md)

            # update measurements
            self.save_processed_data()

    def define_roi(self, data):
        """
        Adds a "selected" attribute to measurements dataframe. The attribute is True for cells that fall within the ROI.

        Args:

            data (pd.DataFrame) - processed measurement data

        """

        if self.include:

            # load ROI vertices
            io = IO()
            path = join(self.subdirs['selection'],'selection.npy')
            roi_vertices = io.read_npy(path)

            # apply mask
            self._apply_roi_vertices(data, self.xykey, roi_vertices)

        else:
            data['selected'] = False


class LayerCorrection:
    """

    Bleedthrough correction related methods for Layer class.

    """

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


class LayerAnnotation:
    """

    Annotation related methods for Layer class.

    """

    def annotate(self):
        """
        Annotate measurement data in place, also labeling boundaries between labeled regions and marking regions in which each label occurs.
        """

        # make sure graph is available
        msg = 'Graph not found. Call the .build_graph() method then try again.'
        assert self.graph is not None, msg

        # make sure annotator is available
        msg = 'Trained annotator not found. Call the .train_annotator() method then try again.'
        assert self.has_trained_annotator, msg

        # apply trained annotator to label distinct celltypes
        self._apply_annotation(self.data)

        # mark boundaries between labeled regions
        self._mark_boundaries(self.data, basis='genotype', max_edges=1)

        # mark regions in which each label is found
        self._apply_concurrency(self.data, basis='genotype')

    def train_annotator(self, attribute,
                        save=False,
                        logratio=True,
                        num_labels=3,
                        **kwargs):
        """
        Train an Annotation model on the measurements in this layer.

        Args:

            attribute (str) - measured attribute used to determine labels

            save (bool) - if True, save model selection routine

            logratio (bool) - if True, weight edges by relative attribute value

            num_labels (int) - number of allowable unique labels

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
        self.annotator = Annotation(attribute, num_labels=num_labels, **kwargs)

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
        data[label] = self.annotator(self.graph, **kwargs)

    def apply_annotation(self, label='genotype', **kwargs):
        """
        Assign labels to cell measurements in place.

        Args:

            label (str) - attribute name for predicted genotype

            kwargs: keyword arguments for Annotator.annotate()

        """
        self._apply_annotation(self.data, label=label, **kwargs)

    @staticmethod
    def _apply_concurrency(data, basis='genotype',
                          min_pop=5,
                          max_distance=10,
                          **kwargs):
        """
        Add boolean 'concurrent_<basis>' field to measurement data for each unique value of <basis> attribute.

        Args:

            data (pd.DataFrame) - processed cell measurement data

            basis (str) - attribute on which concurrency is established

            min_pop (int) - minimum population size for inclusion of cell type

            max_distance (float) - maximum distance threshold for inclusion

            kwargs: keyword arguments for ConcurrencyLabeler

        """

        assert basis in data.columns, 'Attribute {:s} not found.'.format(basis)

        labeler = ConcurrencyLabeler(attribute=basis,
                                    min_pop=min_pop,
                                    max_distance=max_distance,
                                    **kwargs)
        labeler(data)

    def apply_concurrency(self, basis='genotype',
                          min_pop=5,
                          max_distance=10,
                          **kwargs):
        """
        Add boolean 'concurrent_<basis>' field to measurement data for each unique value of <basis> attribute.

        Args:

            basis (str) - attribute on which concurrency is established

            min_pop (int) - minimum population size for inclusion of cell type

            max_distance (float) - maximum distance threshold for inclusion

            kwargs: keyword arguments for ConcurrencyLabeler

        """

        self._apply_concurrency(self.data,
                                basis=basis,
                                min_pop=min_pop,
                                max_distance=max_distance,
                                **kwargs)

    def _mark_boundaries(self, data, basis='genotype', max_edges=0):
        """
        Mark boundaries between cells with disparate labels by assigning a boundary label to all cells that share an edge with another cell with a different label.

        Args:

            data (pd.DataFrame) - processed cell measurement data

            basis (str) - attribute used to define label

            max_edges (int) - maximum number of edges for interior cells

        """

        # make sure graph is available
        msg = 'Graph not found, call .build_graph() method then try again.'
        assert self.graph is not None, msg

        # make sure basis attribute is available
        msg = 'Attribute {:s} not found in measurement data.'.format(basis)
        assert basis in data.columns, msg

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

    def mark_boundaries(self, basis='genotype', max_edges=0):
        """
        Mark boundaries between cells with disparate labels by assigning a boundary label to all cells that share an edge with another cell with a different label.

        Args:

            basis (str) - attribute used to define label

            max_edges (int) - maximum number of edges for interior cells

        """
        self._mark_boundaries(self.data, basis=basis, max_edges=max_edges)

    def show_annotation(self, channel, label,
                        interior_only=False,
                        selection_only=False,
                        cmap=None,
                        figsize=(8, 4),
                        **kwargs):
        """

        Visualize annotation by overlaying <label> attribute on the image of the specified fluoreascence <channel>.

        Args:

            channel (str) - fluorescence channel to visualize

            label (str) - attribute containing cell type labels

            interior_only (bool) - if True, exclude border regions

            selection_only (bool) - if True, only add contours within ROI

            cmap (matplotlib.ListedColorMap) - color scheme for celltype labels

            figsize (tuple) - figure dimensions

            kwargs: keyword arguments for plt.scatter

        Returns:

            fig (matplotlib.Figure)

        """

        assert label in self.data.keys(), 'No {:s} attribute found. Please check to make sure that annotation is complete.'.format(label)

        # create figure and plot images
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=figsize)
        _ = self.get_channel(channel).show(segments=False, ax=ax0)
        _ = self.get_channel(channel).show(segments=False, ax=ax1)

        # build and overlay attribute mask
        mask = self.build_attribute_mask(label,
                                         interior_only=interior_only,
                                         selection_only=selection_only)
        ax1.imshow(mask, cmap=cmap)

        # rectify dimensions
        ax1.set_xlim(*ax0.get_xlim())
        ax1.set_ylim(*ax0.get_ylim())
        plt.tight_layout()

        return fig


class Layer(LayerIO,
            ImageMultichromatic,
            LayerVisualization,
            LayerProperties,
            LayerMeasurement,
            LayerROI,
            LayerCorrection,
            LayerAnnotation):
    """
    Object represents a single imaged layer.

    Attributes:

        measurements (pd.DataFrame) - raw cell measurement data

        data (pd.DataFrame) - processed cell measurement data

        path (str) - path to layer directory

        _id (int) - layer ID, must be an integer value

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

        color_depth (int) - number of fluorescence channels

        num_cells (int) - number of cells detected by segmentation

        bg_key (str) - key for channel used to generate segmentation

        is_segmented (bool) - if True, layer has been segmented

        has_trained_annotator (bool) - if True, layer has a trained annotator

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

        # initialize measurement data
        self.measurements = None
        self.data = None

        # set annotator
        self.annotator = annotator

        # load labels and instantiate image
        self.load_labels()
        super().__init__(im, labels=self.labels)

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
        data = deepcopy(measurements)

        # load and apply selection
        if 'selection' in self.subdirs.keys():
            self.define_roi(data)

        # load and apply correction
        if 'correction' in self.subdirs.keys():
            self.apply_correction(data)

        # annotate measurements
        if self.has_trained_annotator and self.graph is not None:

            # apply trained annotator to label distinct celltypes
            self._apply_annotation(data, label='genotype')

            # mark boundaries between labeled regions
            self._mark_boundaries(data, basis='genotype', max_edges=1)

            # mark regions in which each label is found
            self._apply_concurrency(data, basis='genotype')

        return data

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
