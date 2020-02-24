import warnings
from os.path import join, exists, abspath, isdir
from os import mkdir, remove
from shutil import move, rmtree
from glob import glob
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..utilities import IO
from ..utilities import UserPrompts
from ..annotation import Annotation

from .layers import Layer
from .silhouette_write import WriteSilhouette

# filter numpy warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


class StackIO(WriteSilhouette):
    """ Methods for saving and loading a Stack instance. """

    @staticmethod
    def from_tif(filepath, bit_depth):
        """
        Initialize stack from tif <filepath>.

        Args:

            path (str) - path to tif image file

            bit_depth (int) - bit depth of raw tif (e.g. 12 or 16)

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

        return Stack(path, bit_depth=bit_depth)

    @staticmethod
    def from_silhouette(filepath, bit_depth):
        """
        Initialize stack from silhouette <filepath>.

        Args:

            path (str) - path to silhouette file

            bit_depth (int) - bit depth of raw tif (e.g. 12 or 16)

        Returns:

            stack (flyqma.Stack)

        """

        raise UserWarning('INCOMPLETE METHOD: TIF FILE REQUIRED.')

        path, ext = filepath.rsplit('.', maxsplit=1)
        assert ext.lower() == 'silhouette', 'Silhouette file not found.'
        _id = path.split('/')[-1]

        # make directory
        mkdir(path)
        move(filepath, join(path, '{:s}.silhouette'.format(_id)))

        return Stack(path, bit_depth=bit_depth)

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
        stack = self._pixel_norm(stack, self.bit_depth)
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

        _id (str) - stack ID

        stack (np.ndarray[float]) - 3D RGB image stack

        shape (tuple) - stack dimensions, (depth, X, Y, 3)

        bit_depth (int) - bit depth of raw tif image

        stack_depth (int) - number of layers in stack

        color_depth (int) - number of fluorescence channels in stack

        annotator (Annotation) - object that assigns labels to measurements

        metadata (dict) - stack metadata

        tif_path (str) - path to multilayer RGB tiff file

        layers_path (str) - path to layers directory

        annotator_path (str) - path to annotation directory

    """

    def __init__(self, path, bit_depth=None):
        """
        Initialize stack from stack directory <path>.

        Args:

            path (str) - path to stack directory

            bit_depth (int) - bit depth of raw tif (e.g. 12 or 16). Value will be read from the stack metadata if None is provided. An error is raised if no value is found.

        """

        # strip trailing slashes
        path = path.rstrip('/')

        # check if path is directly to a silhouette file
        if '.silhouette' in path.lower():
            raise ValueError('Please use the Stack.from_silhouette constructor.')
        elif '.tif' in path.lower():
            raise ValueError('Please use the Stack.from_tif constructor.')

        # set path to stack directory
        self._id = path.rsplit('/', maxsplit=1)[-1]
        self.path = abspath(path)
        self.stack = None

        # find image file (defaults to stack name, otherwise first tif found)
        tifs = list(glob(join(path, '*.tif'), recursive=False))
        assert len(tifs) > 0, 'No tif image files found in stack directory.'
        default_path = join(path, '{:s}.tif'.format(self._id))
        if default_path in tifs:
            self.tif_path = default_path
        else:
            self.tif_path = tifs[0]

        # set layer and annotator directory paths
        self.layers_path = join(self.path, 'layers')
        self.annotator_path = join(self.path, 'annotation')

        # initialize stack
        if not self.is_initialized:
            if type(bit_depth) == int:
                self.initialize(bit_depth=bit_depth)
            else:
                self.prompt_initialization()

        # load metadata
        self.load_metadata()

        # if bit_depth was provided, make sure it's consistent
        if bit_depth is None:
            msg = 'bit_depth was neither specified nor found in the metadata.'
            assert 'bits' in self.metadata.keys(), msg
        else:
            msg = 'Specified bit depth is inconsistent with existing metadata.'
            assert self.bit_depth == bit_depth, msg

        # load annotator
        self.annotator = None
        self.load_annotator()

        # reset layer iterator count
        self.count = 0

    def __getitem__(self, layer_id):
        """ Load layer. """
        return self.load_layer(layer_id, graph=True, use_cache=True, full=True)

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

    def prompt_initialization(self):
        """ Ask user whether to initialize all stack directories. """
        msg = '{:s} directory has not been initialized. Do it now?'.format(self.filename)
        user_response = UserPrompts.boolean_prompt(msg)
        if user_response:
            msg = 'Please enter an image bit depth:'
            bit_depth = UserPrompts.integer_prompt(msg)
            if bit_depth is not None:
                self.initialize(bit_depth=bit_depth)
            else:
                raise ValueError('Could not initialize stack because bit depth value was not recognized.')

    @staticmethod
    def _check_if_initialized(path):
        """ Returns True if <path> contains an initialized stack directory. """
        layers_complete = isdir(join(path, 'layers'))
        metadata_complete = exists(join(path, 'metadata.json'))
        if not (layers_complete and metadata_complete):
            return False
        else:
            return True

    @property
    def is_initialized(self):
        """ Returns True if Stack has been initialized. """
        return self._check_if_initialized(self.path)

    @property
    def is_segmented(self):
        """ True if segmentation is complete. """
        return self.aggregate_measurements() is not None

    @property
    def is_annotated(self):
        """ True if annotation is complete. """
        return self.annotator is not None

    @property
    def included(self):
        """ Indices of included layers. """
        return self.get_included_layers()

    @property
    def filename(self):
        """ Stack filename. """
        return self.path.split('/')[-1]

    @property
    def bit_depth(self):
        """ Bit depth of raw image. """
        return self.metadata['bits']

    @property
    def stack_depth(self):
        """ Number of layers in stack. """
        return self.metadata['depth']

    @property
    def color_depth(self):
        """ Number of fluorescence channels in stack. """
        return self.metadata['colordepth']

    @property
    def selector_path(self):
        """ Path to model selection object. """
        return join(self.annotator_path, 'models')

    def get_included_layers(self):
        """ Returns indices of included layers. """
        layers = [self.load_layer(i, graph=False) for i in range(self.stack_depth)]
        return [layer._id for layer in layers if layer.include]

    def initialize(self, bit_depth):
        """
        Initialize stack directory.

        Args:

            bit_depth (int) - bit depth of raw tif (e.g. 12 or 16)

        """

        # make layers directory
        if not exists(self.layers_path):
            mkdir(self.layers_path)

        # load stack (to determine shape), and check bit depth
        raw = self._read_tif(self.tif_path)
        assert raw.max() <= (2**bit_depth), 'Pixels exceed bit_depth.'
        stack = raw / (2**bit_depth)

        # make metadata file
        io = IO()
        self.metadata = dict(bits=bit_depth, depth=stack.shape[0], params={})
        io.write_json(join(self.path, 'metadata.json'), self.metadata)

        # load image
        if self.stack is None:
            self.load_image()

        # initialize layers
        for layer_id in range(self.stack_depth):
            layer_path = join(self.layers_path, '{:d}'.format(layer_id))
            layer = Layer(layer_path)
            layer.initialize()

    def restore_directory(self):
        """ Restore stack directory to original state. """

        dirs = [
            self.layers_path,
            join(self.path, 'annotation'),
            join(self.path, 'metadata.json')]

        for path in dirs:
            if exists(path):
                if isdir(path):
                    rmtree(path)
                else:
                    remove(path)

    def segment(self, channel,
                preprocessing_kws={},
                seed_kws={},
                seg_kws={},
                min_area=250,
                save=True):
        """
        Segment all layers using watershed strategy.

        Args:

            channel (int) - channel index on which to segment image

            preprocessing_kws (dict) - keyword arguments for image preprocessing

            seed_kws (dict) - keyword arguments for seed detection

            seg_kws (dict) - keyword arguments for segmentation

            min_area (int) - threshold for minimum segment size, px

            save (bool) - if True, save measurement data for each layer

        """

        # make sure image is loaded
        assert self.stack is not None, 'Image data not found. Use the load_image() method to read image data into memory.'

        for layer in self:
            _ = layer.segment(channel,
                preprocessing_kws=preprocessing_kws,
                seed_kws=seed_kws,
                seg_kws=seg_kws,
                min_area=min_area)

            # save layer measurements and segmentation
            if save:
                layer.save()

    def train_annotator(self, attribute,
                        save=False,
                        logratio=True,
                        num_labels=3,
                        **kwargs):
        """
        Train an Annotation model on all layers in this stack.

        Args:

            attribute (str) - measured attribute used to determine labels

            save (bool) - if True, save annotator and model selection routine

            logratio (bool) - if True, weight edges by relative attribute value

            num_labels (int) - number of allowable unique labels

            kwargs: keyword arguments for Annotation, including:

                sampler_type (str) - either 'radial', 'neighbors', 'community'

                sampler_kwargs (dict) - keyword arguments for sampler

                min_num_components (int) - minimum number of mixture components

                max_num_components (int) - maximum number of mixture components

                addtl_kwargs: keyword arguments for Classifier

        """

        # make sure measurement data are available
        if not self.is_segmented:
            raise RuntimeError('Measurement data not available, please run segmentation first.')

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
        annotator = Annotation(attribute, num_labels=num_labels, **kwargs)
        selector = annotator.train(*graphs)
        self.annotator = annotator

        # save models
        if save:
            self.save_annotator(data=True)
            selector.save(self.annotator_path)

            # annotate measurements
            for layer in self:
                layer.annotate()
                layer.save_processed_data()

    def aggregate_measurements(self,
                               selected_only=False,
                               exclude_boundary=False,
                               raw=False,
                               use_cache=True):
        """
        Aggregate measurements from each included layer.

        Args:

            selected_only (bool) - if True, exclude cells not marked for inclusion

            exclude_boundary (bool) - if True, exclude cells on clone boundaries

            raw (bool) - if True, aggregate raw measurements

            use_cache (bool) - if True, used available cached measurement data

        Returns:

            data (pd.Dataframe) - measurement data (None if unavailable)

        """

        # load measurements from each included layer
        data = []
        for layer_id in self.included:
            layer = self.load_layer(layer_id,
                                    graph=(not use_cache),
                                    use_cache=use_cache,
                                    full=False)

            # skip layers without measurements
            if not layer.is_segmented:
                continue

            # get raw or processed measurements
            if raw:
                layer_data = layer.measurements
            else:
                layer_data = layer.data

            layer_data['layer'] = layer._id
            data.append(layer_data)
            assert layer_id == layer._id, 'Layer IDs do not match.'

        # return None if no data are found
        if len(data) == 0:
            return None

        # aggregate measurement data
        data = pd.concat(data, join='outer', sort=False)
        data = data.set_index(['layer', 'segment_id'])

        # exclude cells outside the ROI
        if selected_only:
            assert 'selected' in data.columns, 'ROI not defined.'
            data = data[data.selected]

        # exclude cells on the border of labeled regions
        if exclude_boundary:
            assert 'boundary' in data.columns, 'Cannot exclude boundary regions because no regions have been defined. Annotate the stack then try again.'
            data = data[~data.boundary]

        # load manual labels from silhouette
        if exists(self.silhouette_path):
            data = data.join(self.load_silhouette_labels())

        return data

    def load_layer(self, layer_id=0, graph=True, use_cache=True, full=True):
        """
        Load individual layer.

        Args:

            layer_id (int) - layer index

            graph (bool) - if True, load layer graph

            use_cache (bool) - if True, use cached layer measurement data

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
        layer.load(use_cache=use_cache, graph=graph)

        return layer
