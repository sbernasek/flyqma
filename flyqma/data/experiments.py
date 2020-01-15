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

        stack_ids (list) - unique stack ids within experiment

        stack_dirs (dict) - {stack_id: stack_directory} tuples

        count (int) - counter for stack iteration

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
        stack_paths = [p for p in glob(join(self.path, '*')) if isdir(p)]
        get_stack_id = lambda x: x.rsplit('/', maxsplit=1)[-1]
        self.stack_dirs = {get_stack_id(p): p for p in stack_paths}
        self.stack_ids = sorted(self.stack_dirs.keys())

        # reset stack iterator count
        self.count = 0

    def __getitem__(self, stack_id):
        """ Load stack. """
        return self.load_stack(stack_id, full=False)

    def __iter__(self):
        """ Iterate across stacks. """
        self.count = 0
        return self

    def __next__(self):
        """ Return next stack. """
        if self.count < len(self.stacks):
            stack_id = self.stack_ids[self.count]
            stack = self.__getitem__(stack_id)
            self.count += 1
            return stack
        else:
            raise StopIteration

    def load_stack(self, stack_id, full=False, **kwargs):
        """
        Load 3D RGB image stack.

        Args:

            stack_id (str or int) - stack to be loaded

            full (bool) - if True, load full 3D image from tif file

        Returns:

            stack (Stack)

        """
        stack = Stack(self.stack_dirs[str(stack_id)], **kwargs)
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

            data (pd.Dataframe) - curated cell measurement data

        """

        # load measurements from each stack in the experiment
        data = []
        for stack_id in self.stack_ids:
            stack = self.load_stack(stack_id, full=False)
            measurements = stack.aggregate_measurements(raw=raw)

            # add stack index
            measurements['stack'] = stack._id
            measurements = measurements.set_index('stack', append=True)
            measurements = measurements.reorder_levels([2,0,1])

            data.append(measurements)
            assert stack_id == stack._id, 'Stack IDs do not match.'

        # aggregate measurements
        data = pd.concat(data, join='outer', sort=False)

        # exclude cells that were not marked for inclusion
        if selected_only:
            data = data[data.selected]

        # exclude cells on clone boundaries
        if exclude_boundary:
            data = data[~data.boundary]

        return data
