from os.path import join, abspath, isdir
from glob import glob
import numpy as np
import pandas as pd

from ..utilities import UserPrompts

from .stacks import Stack


class Experiment:
    """
    Object represents a collection of 3D RGB image stacks collected under the same experimental conditions.

    Attributes:

        path (str) - path to experiment directory

        _id (str) - name of experiment

        stack_ids (list of str) - unique stack ids within experiment

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

        # check if stacks have been initialized, if not prompt user
        if not self.is_initialized:
            self.prompt_initialization()

    def __getitem__(self, stack_id):
        """ Load stack. """
        return self.load_stack(stack_id, full=False)

    def __iter__(self):
        """ Iterate across stacks. """
        self.count = 0
        return self

    def __next__(self):
        """ Return next stack. """
        if self.count < len(self.stack_ids):
            stack_id = self.stack_ids[self.count]
            stack = self.__getitem__(stack_id)
            self.count += 1
            return stack
        else:
            raise StopIteration

    @property
    def is_initialized(self):
        """ Returns True if Experiment has been initialized. """
        for stack_dir in self.stack_dirs.values():
            if not Stack._check_if_initialized(stack_dir):
                return False
        return True

    def prompt_initialization(self):
        """ Ask user whether to initialize all stack directories. """
        msg = 'Incomplete stack directories found. Initialize them?'
        user_response = UserPrompts.boolean_prompt(msg)
        if user_response:
            msg = 'Please enter an image bit depth:'
            bit_depth = UserPrompts.integer_prompt(msg)
            if bit_depth is not None:
                self.initialize(bit_depth)
            else:
                raise ValueError('User response not recognized, stacks have not been initialized.')

    def initialize(self, bit_depth):
        """
        Initialize a collection of image stacks.

        Args:

            bit_depth (int) - bit depth of raw tif (e.g. 12 or 16). Value will be read from the stack metadata if None is provided. An error is raised if no value is found.

        """
        for stack_id in self.stack_ids:
            _ = self.load_stack(stack_id, full=False, bit_depth=bit_depth)

    def load_stack(self, stack_id, full=False, **kwargs):
        """
        Load 3D RGB image stack.

        Args:

            stack_id (str or int) - desired stack

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
                               raw=False,
                               use_cache=True):
        """
        Aggregate measurements from each stack.

        Args:

            selected_only (bool) - if True, exclude cells outside the ROI

            exclude_boundary (bool) - if True, exclude cells on the border of labeled regions

            raw (bool) - if True, use raw measurements from included discs

            use_cache (bool) - if True, used available cached measurement data

        Returns:

            data (pd.Dataframe) - curated cell measurement data, which is None if no measurement data are found

        """

        # load measurements from each stack in the experiment
        data = []
        for stack_id in self.stack_ids:
            stack = self.load_stack(stack_id, full=False)
            measurements = stack.aggregate_measurements(
                selected_only=selected_only,
                exclude_boundary=exclude_boundary,
                raw=raw,
                use_cache=use_cache)

            if measurements is None:
                continue

            # add stack index
            measurements['stack'] = stack._id
            measurements = measurements.set_index('stack', append=True)
            measurements = measurements.reorder_levels([2,0,1])

            data.append(measurements)
            assert stack_id == stack._id, 'Stack IDs do not match.'

        # return None if no data are found
        if len(data) == 0:
            return None

        # aggregate measurements
        data = pd.concat(data, join='outer', sort=False)

        # exclude cells that were not marked for inclusion
        if selected_only:
            data = data[data.selected]

        # exclude cells on clone boundaries
        if exclude_boundary:
            data = data[~data.boundary]

        return data
