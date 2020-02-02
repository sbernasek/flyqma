from unittest import TestCase
from os.path import join, exists

from flyqma.data import Experiment, Stack
from .test_io import TestPaths


class TestExperiment(TestPaths):
    """
    Tests for Experiment class.
    """

    @classmethod
    def setUpClass(cls):
        """ Initialize test instance of Experiment. """
        cls.experiment = Experiment(cls.exp_path)
        cls.experiment.initialize(bit_depth=cls.bit_depth)

    @classmethod
    def tearDownClass(cls):
        """ Restore stack directories to original state. """
        for stack in cls.experiment:
            stack.restore_directory()

    def test01_load_stack(self):
        """ Load Stack instance from Experiment. """
        stack = self.experiment.load_stack(self.experiment.stack_ids[0])
        self.assertTrue(isinstance(stack, Stack))
