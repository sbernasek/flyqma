from os.path import join, exists, abspath, isdir
from os import mkdir
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

        # load 3D RGB tif
        self.stack = self._read_tif(self.tif_path) / (2**self.metadata['bits'])

        # set stack shape
        self.metadata['depth'] = self.stack.shape[0]

    @staticmethod
    def _read_tif(path, bits=12):
        """
        Read 3D RGB tif file.

        Args:

            bits (int) - tif resolution

        Note: images are flipped from BGR to RGB

        """
        io = IO()
        stack = io.read_tiff(path)
        if len(stack.shape) == 3:
            stack = stack.reshape(1, *stack.shape)
        stack = np.swapaxes(stack, 1, 2)
        stack = np.swapaxes(stack, 2, 3)
        return stack[:, :, :, ::-1]


class Stack(StackIO):
    """
    Object represents a 3D RGB image stack.

    Attributes:

        path (str) - path to stack directory

        _id (int) - stack ID

        stack (np.ndarray[float]) - 3D RGB image stack

        shape (tuple) - stack dimensions, (depth, X, Y, 3)

        depth (int) - number of layers in stack

        annotator (Annotation) - object that assigns labels to measurements

        metadata (dict) - stack metadata

        tif_path (str) - path to multilayer RGB tiff file

        layers_path (str) - path to layers directory

        annotator_path (str) - path to annotation directory

    """

    def __init__(self, path):
        """
        Initialize stack.

        Args:

            path (str) - path to stack directory

        """

        # set path to stack directory
        self.path = abspath(path)
        self.stack = None

        # set paths
        self._id = int(path.rsplit('/', maxsplit=1)[-1])
        self.tif_path = join(path, '{:d}.tif'.format(self._id))
        self.layers_path = join(self.path, 'layers')
        self.annotator_path = join(self.path, 'annotation')

        # initialize stack if layers directory doesn't exist
        if not isdir(self.layers_path):
            self.initialize()

        # load metadata
        self.load_metadata()

        # load annotator
        self.annotator = None
        self.load_annotator()

        # store indices of included layers
        self.included = self.get_included_layers()

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
    def depth(self):
        """ Number of layers in stack. """
        return self.metadata['depth']

    def get_included_layers(self):
        """ Returns indices of included layers. """
        layers = [self.load_layer(i, graph=False) for i in range(self.depth)]
        return [layer._id for layer in layers if layer.include]

    def initialize(self, bits=12):
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
        metadata = dict(bits=bits, depth=stack.shape[0], params={})
        io.write_json(join(self.path, 'metadata.json'), metadata)

        # load image
        if self.stack is None:
            self.load_image()

        # initialize layers
        for layer_id in range(self.depth):
            layer_path = join(self.layers_path, '{:d}'.format(layer_id))
            layer = Layer(layer_path)
            layer.initialize()

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
