from os.path import join, exists, abspath, isdir
from os import mkdir
from glob import glob
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..utilities.io import IO
from ..annotation.classifiers import CellClassifier
from .layers import Layer


class Stack:
    """
    Object represents a 3D RGB image stack.

    Attributes:
    path (str) - path to stack directory
    _id (int) - stack ID
    stack (np.ndarray[float]) - 3D RGB image stack
    shape (tuple) - stack dimensions, (depth, X, Y, 3)
    depth (int) - number of layers in stack
    classifier (CellClassifier) - callable cell classifier
    metadata (dict) - stack metadata
    tif_path (str) - path to multilayer RGB tiff file
    layers_path (str) - path to layers directory
    classifier_path (str) - path to cell classifier directory
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
        self.classifier_path = join(self.path, 'cell_classifier')

        # initialize stack if layers directory doesn't exist
        if not isdir(self.layers_path):
            self.initialize()

        # load metadata and classifier
        self.load_metadata()
        self.load_classifier()

        # reset layer iterator count
        self.count = 0

    def __getitem__(self, layer_id):
        """ Load layer. """
        return self.load_layer(layer_id)

    def __iter__(self):
        """ Iterate across layers. """
        self.count = 0
        return self

    def __next__(self):
        """ Return next layer. """
        if self.count < self.depth:
            layer = self.__getitem__(self.count)
            self.count += 1
            return layer
        else:
            raise StopIteration

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

    def load_metadata(self):
        """
        Load available metadata.
        """
        metadata_path = join(self.path, 'metadata.json')
        if exists(metadata_path):
            io = IO()
            self.metadata = io.read_json(metadata_path)
            self.depth = self.metadata['depth']

    def load_classifier(self):
        """ Load cell classifier from cell_classifier directory. """
        if exists(self.classifier_path):
            self.classifier = CellClassifier.load(self.classifier_path)

    def load_image(self):
        """ Load 3D image from tif file. """

        # load 3D RGB tif
        self.stack = self._read_tif(self.tif_path) / (2**self.metadata['bits'])

        # set stack shape
        self.depth = self.stack.shape[0]
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

    def load_layer(self, layer_id=0, full=True):
        """
        Load individual layer.

        Args:
        layer_id (int) - layer index
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
        layer = Layer(layer_path, im, self.classifier)

        # load layer
        layer.load()

        return layer

    def save_metadata(self):
        """ Save metadata. """
        io = IO()
        io.write_json(join(self.path, 'metadata.json'), self.metadata)

    def aggregate_measurements(self, raw=False):
        """
        Aggregate measurements from each layer.

        Args:
        raw (bool) - if True, aggregate raw measurements from included discs

        Returns:
        data (pd.Dataframe) - processed cell measurement data
        """

        # load measurements from each included layer
        data = []
        for layer_id in range(self.depth):
            layer = self.load_layer(layer_id, full=False)
            if layer.include == True:

                # get raw or processed measurements
                if raw:
                    layer_data = layer.measurements
                else:
                    layer_data = layer.data

                layer_data['layer'] = layer._id
                data.append(layer_data)

        # aggregate measurement data
        data = pd.concat(data, join='inner')

        return data
