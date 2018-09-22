from os.path import join, abspath, isdir
from glob import glob
import numpy as np
import pandas as pd

from .stacks import Stack


class Experiment:
    """
    Object represents a collection of 3D RGB image stacks collected under the same experimental conditions.

    Attributes:
    path (str) - path to experiment directory
    _id (int) - experiment ID
    size (int) - number of stacks in experiment
    df (pd.DataFrame) - curated cell contour measurements
    """

    def __init__(self, path):
        """
        Instantiate experiment object.

        Args:
        path (str) - directory with subdirectories of 3D RGB image stacks
        """

        # set path to experiment directory
        self.path = abspath(path)

        # set experiment ID
        self._id = path.split('/')[-1]

        # set stack paths
        self.stacks = [p for p in glob(join(self.path, '*[0-9]')) if isdir(p)]

        # set experiment size
        self.size = len(self.stacks)

        # reset stack iterator count
        self.count = 0

    def __getitem__(self, stack_ind):
        """ Load stack. """
        return self.load_stack(stack_ind, full=False)

    def __iter__(self):
        """ Iterate across stacks. """
        self.count = 0
        return self

    def __next__(self):
        """ Return next stack. """
        if self.count < self.size:
            stack = self.__getitem__(self.count)
            self.count += 1
            return stack
        else:
            raise StopIteration

    def load_stack(self, stack_ind, full=False):
        """
        Load 3D RGB image stack.

        Args:
        stack_ind (int) - stack index in list of paths
        full (bool) - if True, load full 3D image from tif file

        Returns:
        stack (Stack)
        """
        stack = Stack(self.stacks[stack_ind])
        if full:
            stack.load_image()
        return stack

    def aggregate_measurements(self,
                               selected_only=False,
                               exclude_boundary=False,
                               raw=False):
        """
        Aggregate measurements from each stack.

        Args:
        selected_only (bool) - if True, exclude cells not marked for inclusion
        exclude_boundary (bool) - if True, exclude cells on clone boundaries
        raw (bool) - if True, aggregate raw measurements from included discs

        Returns:
        measurements (pd.Dataframe) - curated cell measurement data
        """

        # load measurements from each stack in the experiment
        measurements = []
        for stack_ind in range(self.size):
            stack = self.load_stack(stack_ind, full=False)
            df = stack.aggregate_measurements(raw=raw)
            df['stack'] = stack._id
            measurements.append(df)

        # aggregate measurements
        measurements = pd.concat(measurements, join='inner')

        # exclude cells that were not marked for inclusion
        if selected_only:
            measurements = measurements[measurements.selected]

        # exclude cells on clone boundaries
        if exclude_boundary:
            measurements = measurements[~measurements.boundary]

        return measurements
