import warnings
from os.path import join, exists, abspath, isdir
from os import mkdir
from shutil import move
from glob import glob
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..utilities import IO
from ..annotation import Annotation

from .layers import Layer
from .silhouette import SilhouetteIO


class StackIO(SilhouetteIO):
    """ Methods for saving and loading a Stack instance. """

    @staticmethod
    def from_tif(filepath, bits=16):
        """
        Initialize stack from tif <filepath>.

        Args:

            path (str) - path to tif image file

            bits (int) - bit depth

        Returns:

            stack (flyqma.Stack)

        """

        path, ext = filepath.rsplit('.', maxsplit=1)
        if ext.lower() != 'tif':
            raise ValueError('TIF extension not found.')
        _id = path.split('/')[-1]

        # make directory
        mkdir(path)
        move(filepath, join( path, '{:s}.tif'.format(_id)))

        return Stack(path, bits=bits)

    @staticmethod
    def from_silhouette(filepath, bits=16):
        """
        Initialize stack from silhouette <filepath>.

        Args:

            path (str) - path to silhouette file

            bits (int) - bit depth

        Returns:

            stack (flyqma.Stack)

        """

        raise UserWarning('INCOMPLETE FUNCTIONALITY: TIF FILE REQUIRED.')

        path, ext = filepath.rsplit('.', maxsplit=1)
        if ext.lower() != 'silhouette':
            raise ValueError('Silhouette extension not recognized.')
        _id = path.split('/')[-1]

        # make directory
        mkdir(path)
        move(filepath, join( path, '{:s}.silhouette'.format(_id)))

        return Stack(path, bits=bits)

    def save(self):
        """ Save stack metadata and annotator. """
        self.save_metadata()
        if self.annotator is not None:
            self.save_annotator()

    def save_metadata(self):
        """ Save metadata. """
        io = IO()
        io.write_json(join(self.path, 'metadata.json'), self.metadata)

    def save_annotator(self, data=True):
        """ Save annotator to annotation directory. """
        if not isdir(self.annotator_path):
            mkdir(self.annotator_path)
        self.annotator.save(self.annotator_path, data=data)

    def load_metadata(self):
        """ Load available metadata. """
        metadata_path = join(self.path, 'metadata.json')
        if exists(metadata_path):
            io = IO()
            self.metadata = io.read_json(metadata_path)

    def load_annotator(self):
        """ Load annotator from annotation directory. """
        if exists(self.annotator_path):
            self.annotator = Annotation.load(self.annotator_path)

    def load_image(self):
        """ Load 3D image from tif file. """

        # load tif, normalize pixel intensities, and convert to NWHC format
        stack = self._read_tif(self.tif_path)
        stack = self._pixel_norm(stack, self.metadata['bits'])
        stack = self._to_NWHC(stack)
        self.stack = stack

        # set stack shape
        self.metadata['depth'] = self.stack.shape[0]
        self.metadata['colordepth'] = self.stack.shape[-1]

    @staticmethod
    def _read_tif(path):
        """ Read tif from <path>. """
        return IO().read_tiff(path)

    @staticmethod
    def _pixel_norm(pixels, bitdepth=12):
        """ Normalize <pixels> intensities by <bitdepth>. """
        return pixels / (2**bitdepth)

    @staticmethod
    def _reorder_channels(stack, idx):
        """ Reorder channels in <stack> according to <idx>. """
        return stack[:, :, :, idx]

    @staticmethod
    def _to_NWHC(stack):
        """
        Convert image to NWHC format. Native format is automatically detected using some heuristics regarding image size relative to channel depth.

        Args:

            stack (np.ndarray[float]) - original image stack

        Returns:

            stack_NWHC (np.ndarray[float]) - image stack in NWHC format

        """

        # if only one layer is provided, append depth dimension (N)
        if len(stack.shape) == 3:
            stack = stack.reshape(1, *stack.shape)

        # determine channel dimension (C) - assumes it's smaller than W & H
        c_dim = np.argmin(stack.shape[1:]) + 1

        # swap axes until C is the last dimension
        while c_dim != 3:
            stack = np.swapaxes(stack, c_dim, c_dim+1)
            c_dim += 1

        return stack


class Stack(StackIO):
    """
    Object represents a 3D RGB image stack.

    Attributes:

        path (str) - path to stack directory

        _id (str or int) - stack ID

        stack (np.ndarray[float]) - 3D RGB image stack

        shape (tuple) - stack dimensions, (depth, X, Y, 3)

        depth (int) - number of layers in stack

        annotator (Annotation) - object that assigns labels to measurements

        metadata (dict) - stack metadata

        tif_path (str) - path to multilayer RGB tiff file

        layers_path (str) - path to layers directory

        annotator_path (str) - path to annotation directory

    """

    def __init__(self, path, bits=16):
        """
        Initialize stack from stack directory <path>.

        Args:

            path (str) - path to stack directory

            bits (int) - bit depth

        """

        # strip trailing slashes
        path = path.rstrip('/')

        # check if path is directly to a silhouette file
        if '.silhouette' in path.lower():
            raise ValueError('This is a silhouette file, use the Stack.from_silhouette constructor.')
        elif '.tif' in path.lower():
            raise ValueError('This is a tif file, use the Stack.from_tif constructor.')

        # set path to stack directory
        self._id = path.rsplit('/', maxsplit=1)[-1]
        self.path = abspath(path)
        self.stack = None

        # set paths
        self.tif_path = join(path, '{:s}.tif'.format(self._id))
        self.layers_path = join(self.path, 'layers')
        self.annotator_path = join(self.path, 'annotation')

        # initialize stack if layers directory doesn't exist
        if not isdir(self.layers_path):
            self.initialize(bits=bits)

        # load metadata
        self.load_metadata()

        # load annotator
        self.annotator = None
        self.load_annotator()

        # reset layer iterator count
        self.count = 0

    def __getitem__(self, layer_id):
        """ Load layer. """
        return self.load_layer(layer_id, graph=True, process=False, full=True)

    def __iter__(self):
        """ Iterate across included layers. """
        self.count = 0
        return self

    def __next__(self):
        """ Return next included layer. """
        if self.count < len(self.included):
            layer_id = self.included[self.count]
            layer = self.__getitem__(layer_id)
            self.count += 1
            return layer
        else:
            raise StopIteration

    @property
    def included(self):
        """ Indices of included layers. """
        return self.get_included_layers()

    @property
    def depth(self):
        """ Number of layers in stack. """
        return self.metadata['depth']

    @property
    def colordepth(self):
        """ Number of color channels in stack. """
        return self.metadata['colordepth']

    @property
    def selector_path(self):
        """ Path to model selection object. """
        return join(self.annotator_path, 'models')

    def get_included_layers(self):
        """ Returns indices of included layers. """
        layers = [self.load_layer(i, graph=False) for i in range(self.depth)]
        return [layer._id for layer in layers if layer.include]

    def initialize(self, bits=16):
        """
        Initialize stack directory.

        Args:

            bits (int) - tif resolution

        """

        # make layers directory
        if not exists(self.layers_path):
            mkdir(self.layers_path)

        # load stack (to determine shape)
        stack = self._read_tif(self.tif_path) / (2**bits)

        # make metadata file
        io = IO()
        self.metadata = dict(bits=bits, depth=stack.shape[0], params={})
        io.write_json(join(self.path, 'metadata.json'), self.metadata)

        # load image
        if self.stack is None:
            self.load_image()

        # initialize layers
        for layer_id in range(self.depth):
            layer_path = join(self.layers_path, '{:d}'.format(layer_id))
            layer = Layer(layer_path)
            layer.initialize()

    def segment(self, channel,
                preprocessing_kws={},
                seed_kws={},
                seg_kws={},
                min_area=250):
        """
        Segment all layers using watershed strategy.

        Args:

            channel (int) - channel index on which to segment image

            preprocessing_kws (dict) - keyword arguments for image preprocessing

            seed_kws (dict) - keyword arguments for seed detection

            seg_kws (dict) - keyword arguments for segmentation

            min_area (int) - threshold for minimum segment size, px

        """

        for layer in self:
            _ = layer.segment(channel,
                preprocessing_kws=preprocessing_kws,
                seed_kws=seed_kws,
                seg_kws=seg_kws,
                min_area=min_area)

            # save layer measurements and segmentation
            layer.save()

    def train_annotator(self, attribute, save=False, logratio=True, **kwargs):
        """
        Train an Annotation model on all layers in this stack.

        Args:

            attribute (str) - measured attribute used to determine labels

            save (bool) - if True, save annotator and model selection routine

            logratio (bool) - if True, weight edges by relative attribute value

            kwargs: keyword arguments for Annotation, including:

                sampler_type (str) - either 'radial', 'neighbors', 'community'

                sampler_kwargs (dict) - keyword arguments for sampler

                min_num_components (int) - minimum number of mixture components

                max_num_components (int) - maximum number of mixture components

                addtl_kwargs: keyword arguments for Classifier

        """

        # build graph for each layer
        graphs = []
        for layer_id in self.included:
            layer = self.load_layer(layer_id, False, False, False)
            layer.build_graph(attribute, logratio=logratio)
            graphs.append(layer.graph)

            # save graph metadata
            if save:
                layer.save_metadata()

        # train annotator
        annotator = Annotation(attribute, **kwargs)
        selector = annotator.train(*graphs)
        self.annotator = annotator

        # save models
        if save:
            self.save_annotator(data=True)
            selector.save(self.annotator_path)

            # annotate measurements
            for layer in self:
                layer.annotate(layer.data)
                layer.save_processed_data()

    def aggregate_measurements(self, raw=False, process=False):
        """
        Aggregate measurements from each included layer.

        Args:

            raw (bool) - if True, aggregate raw measurements

            process (bool) - if True, apply processing to raw measurements

        Returns:

            data (pd.Dataframe) - processed cell measurement data

        """

        # load measurements from each included layer
        data = []
        for layer_id in self.included:
            layer = self.load_layer(layer_id,
                                    graph=process,
                                    process=process,
                                    full=False)

            # get raw or processed measurements
            if raw:
                layer_data = layer.measurements
            else:
                layer_data = layer.data

            layer_data['layer'] = layer._id
            data.append(layer_data)

        # aggregate measurement data
        data = pd.concat(data, join='outer', sort=False)
        data = data.set_index(['layer', 'segment_id'])

        # load manual labels from silhouette
        if exists(self.silhouette_path):
            data = data.join(self.load_silhouette_labels())

        return data

    def load_layer(self, layer_id=0, graph=True, process=False, full=True):
        """
        Load individual layer.

        Args:

            layer_id (int) - layer index

            graph (bool) - if True, load layer graph

            process (bool) - if True, re-process the layer measurement data

            full (bool) - if True, load fully labeled RGB image

        Returns:

            layer (Layer)

        """

        # define layer path
        layer_path = join(self.layers_path, '{:d}'.format(layer_id))

        # if performing light load, don't pass image
        if full and self.stack is not None:
            im = self.stack[layer_id, :, :, :]
        else:
            im = None

        # instantiate layer
        layer = Layer(layer_path, im, self.annotator)

        # load layer
        layer.load(process=process, graph=graph)

        return layer
