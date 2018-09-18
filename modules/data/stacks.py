import os
import json
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import mean, standard_deviation
from collections import Counter
from modules.io import IO
from modules.image import MultichannelImage
from modules.contours import Contours
from modules.kde import KDE
from modules.rbf import RBF
from modules.masking import ForegroundMask, RBFMask
from modules.segmentation import Segmentation
from modules.graphs import WeightedGraph
from modules.classification import CellClassifier
from modules.annotation import Annotation


from os.path import join, exists
from os import mkdir


class Stack:
    """
    Object represents a 3D RGB image stack.

    Attributes:
    path (str) - path to stack directory
    _id (int) - stack ID
    metadata (dict) - stack metadata
    tif_path (str) - path to multilayer RGB tiff file
    layers_path (str) - path to layers directory
    stack (np.ndarray[float]) - 3D RGB image stack
    shape (tuple) - stack dimensions, (depth, X, Y, 3)
    depth (int) - number of layers in stack
    df (pd.DataFrame) - cell contour measurements
    """

    def __init__(self, path):
        """
        Initialize stack.

        Args:
        path (str) - path to stack directory
        """

        # set path to stack directory
        self.path = path

        # set paths
        self._id = int(path.rsplit('/', maxsplit=1)[-1])
        self.tif_path = join(path, '{:d}.tif'.format(self._id))
        self.layers_path = join(self.path, 'layers')
        self.contours_path = join(self.path, 'contours.json')

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

        # make metadata file
        io = IO()
        metadata = dict(bits=bits, params={})
        io.write_json(join(self.path, 'metadata.json'), metadata)

        # make measurements file
        io.write_json(self.contours_path, {})

    def save(self):
        """
        Save measurements.
        """

        # instantiate IO
        io = IO()

        # save contours to json
        contours = self.df.to_json()
        io.write_json(self.contours_path, contours)

        return stack_path

    def load(self):
        """
        Load stack.
        """

        # load metadata
        io = IO()
        self.metadata = io.read_json(join(self.path, 'metadata.json'))

        # load 3D RGB tif
        self.stack = self._read_tif(self.tif_path) / (2**self.metadata['bits'])

        # set stack shape
        self.shape = self.stack.shape
        self.depth = self.shape(0)

        # load measurements
        self.df = self.load_measurements()

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

    def get_layer(self, layer_id=0):
        """
        Instantiate single layer.

        Args:
        layer_id (int) - layer index

        Returns:
        layer (clones.data.layers.Layer)
        """
        im = self.stack[layer_id, :, :, :]

        layer_path = join(self.layers_path, '{:d}'.format(layer_id))

        layer = Layer(im, layer_id=layer_id, stack_path=self.path)
        return layer

    def load_layer(self, layer_id=0):
        """
        Load single layer.

        Args:
        layer_id (int) - layer index

        Returns:
        layer (clones.data.layers.Layer)
        """
        layer_path = join(self.layers_path, '{:d}'.format(layer_id))
        im = self.stack[layer_id, :, :, :]
        return Layer.load(im, layer_path)

    def load_metadata(self):
        """
        Load metadata from segmentation file.

        Returns:
        metadata (dict) - stack metadata
        """
        metadata_path = os.path.join(self.path, 'metadata.json')
        io = IO()
        return io.read_json(metadata_path)

    def load_measurements(self):
        """
        Load contour measurements from contours file.

        Returns:
        measurements (pd.Dataframe) - contour measurements
        """
        io = IO()
        return pd.read_json(io.read_json(self.contours_path))

""""" UPDATE BELOW THIS LINE

    def segment(self, bg='b',
                    segmentation_kw={},
                    fg_kw={},
                    annotation_kw={},
                    save_kw={},
                    save=False):
        """ Segment and annotate all layers in stack. """

        # create directory and save parameters
        self.params = dict(segmentation_kw=segmentation_kw, fg_kw=fg_kw, annotation_kw=annotation_kw)

        if save:
            self.create_directory()

        # segment and annotate each layer
        dfs = []
        for layer_id, im in enumerate(self.stack):
            layer = Layer(im, layer_id=layer_id, stack_path=self.path)
            layer.segment(bg=bg, **segmentation_kw)
            layer.compile_dataframe()

            # save layer segmentation
            if save:
                _ = layer.save(**save_kw)
            dfs.append(layer.df)

        # compile dataframe
        self.df = pd.concat(dfs).reset_index(drop=True)
        if save:
            _ = self.save(self.path)










