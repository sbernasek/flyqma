import os
import glob
import json
import pandas as pd
from modules.io import IO
from modules.stacks import Stack


class Experiment:

    def __init__(self, path):
        self.path = path
        self.genotype = path.split('/')[-1]
        self.condition = path.split('/')[-3]
        disc_names, discs = self.compile_discs()
        self.disc_names = disc_names
        self.discs = discs

    def get_segmentation_paths(self):
        paths = glob.glob(os.path.join(self.path, '*[0-9]'))
        return [p for p in paths if os.path.isdir(p)]

    def get_image_paths(self):
        return glob.glob(os.path.join(self.path, '*[0-9].tif'))

    def compile_discs(self):
        discs = {}
        for path in self.get_segmentation_paths():
            disc = Disc(path, genotype=self.genotype)
            discs[disc.disc_id] = disc
        disc_names = [k for k in sorted(discs.keys())]
        return disc_names, discs

    def compile_measurements(self):
        measurements = []
        for disc in self.discs.values():
            measurements.append(disc.load_measurements())
        measurements = pd.concat(measurements)
        measurements.reset_index(drop=True, inplace=True)
        return measurements


class Disc:

    def __init__(self, path, genotype=None):

        self.path = path

        base, disc_id = path.rsplit('/', maxsplit=1)
        self.disc_id = int(disc_id)

        if genotype is None:
            _, genotype = base.rsplit('/', maxsplit=1)
        self.genotype = genotype

        # load metadata
        self.metadata = self.load_metadata(path)

    @staticmethod
    def load_metadata(path):
        """
        Load segmentation metadata.

        Args:
        path (str) - path to segmentation directory
        """

        io = IO()
        metadata = io.read_json(os.path.join(path, 'metadata.json'))
        return metadata

    def parse_metadata(self):
        """ Unpack metadata from segmentation. """
        self.image_path = self.metadata['path']
        self.bits = self.metadata['bits']
        self.params = self.metadata['params']

    @staticmethod
    def _load_measurements(path):
        """ Load contour data as dataframe. """
        contours_path = os.path.join(path, 'contours.json')
        with open(contours_path, 'r') as f:
            df = pd.read_json(json.load(f))
        return df

    def load_measurements(self):
        """ Load measurement dataframe. """
        df = self._load_measurements(self.path)
        df['genotype'] = self.genotype
        df['disc_id'] = self.disc_id
        return df

    def load_stack(self):
        """ Load Stack instance. """
        return self._load_stack(self.path)

    @staticmethod
    def _load_stack(path):
        """ Load Stack instance from segmentation directory. """
        return Stack.from_segmentation(path)


