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


class Stack:
    """
    Object represents a 3D RGB image stack.

    Attributes:

    """

    def __init__(self, tif_path, bits=2**12, params={}):
        disc_name = tif_path.split('/')[-1].split('.')[0]
        self.disc_name = disc_name
        self.genotype_path = tif_path.rsplit('/', maxsplit=1)[0]
        self.tif_path = tif_path
        self.bits = bits
        self.stack = self.read_tif(tif_path) / bits
        self.shape = self.stack.shape
        self.depth = self.shape[0]
        self.params = params
        self.df = None
        self.path = os.path.join(self.genotype_path, disc_name)
        self.count = 0

    def __getitem__(self, ind):
        return self.load_layer(ind)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < self.depth:
            layer = self.__getitem__(self.count)
            self.count += 1
            return layer
        else:
            raise StopIteration

    @staticmethod
    def read_tif(path):
        """ Loads raw tif. Note: images are flipped from BGR to RGB """
        io = IO()
        stack = io.read_tiff(path)
        if len(stack.shape) == 3:
            stack = stack.reshape(1, *stack.shape)
        stack = np.swapaxes(stack, 1, 2)
        stack = np.swapaxes(stack, 2, 3)
        return stack[:, :, :, ::-1]

    def get_layer(self, layer_id=0):
        """ Instantiate MultiChannel image of single layer. """
        im = self.stack[layer_id, :, :, :]
        layer = Layer(im, layer_id=layer_id, stack_path=self.path)
        return layer

    def load_layer(self, layer_id=0):
        """ Load segmented MultiChannel image of single layer. """
        layer_path = os.path.join(self.path, '{:d}'.format(layer_id))
        im = self.stack[layer_id, :, :, :]
        return Layer.load(im, layer_path)

    def load_metadata(self):
        """ Load metadata from segmentation file. """
        metadata_path = os.path.join(self.path, 'metadata.json')
        io = IO()
        return io.read_json(metadata_path)

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
            layer.set_foreground(**fg_kw)
            layer.annotate(**annotation_kw)

            # save layer segmentation
            if save:
                _ = layer.save(**save_kw)
            dfs.append(layer.df)

        # compile dataframe
        self.df = pd.concat(dfs).reset_index(drop=True)
        if save:
            _ = self.save(self.path)

    def create_directory(self):
        """ Create directory for segmentation results. """
        io = IO()
        io.make_dir(self.path, force=True)

    def save(self, stack_path):
        """ Save segmentation parameters and results. """

        # instantiate IO
        io = IO()

        # save metadata to json
        metadata = dict(path=self.tif_path,
                        bits=self.bits,
                        params=self.params)
        io.write_json(os.path.join(stack_path, 'metadata.json'), metadata)

        # save contours to json
        contours = self.df.to_json()
        io.write_json(os.path.join(stack_path, 'contours.json'), contours)

        return stack_path

    @staticmethod
    def from_segmentation(path):
        """ Load segmented Stack from file. """

        io = IO()

        # load metadata
        metadata_path = os.path.join(path, 'metadata.json')
        metadata = io.read_json(metadata_path)

        # parse metadata
        saved_loc, saved_path = metadata['path'].rsplit('clones/', maxsplit=1)
        current_loc = path.rsplit('clones', maxsplit=1)[0] + 'clones'
        tif_path = os.path.join(current_loc, saved_path)

        bits = metadata['bits']
        params = metadata['params']

        # instantiate stack
        stack = Stack(tif_path, bits=bits, params=params)
        stack.path = path

        # load measurements
        contours_path = os.path.join(path, 'contours.json')
        stack.df = pd.read_json(io.read_json(contours_path))

        return stack







