from unittest.mock import patch
from os.path import join, exists

from flyqma.data import Experiment, Stack
from .test_io import TestPaths


class Test04_Experiment(TestPaths):
    """
    Tests for Experiment instance.
    """

    @classmethod
    def setUpClass(cls):
        """ Initialize test instance of Experiment. """
        with patch('builtins.input', side_effect=['yes', 12]):
            experiment = Experiment(cls.experiment_path)
        cls.experiment = experiment

    @classmethod
    def tearDownClass(cls):
        """ Restore stack directories to original state. """
        for stack in cls.experiment:
            stack.restore_directory()

    def test01_load_stack(self):
        """ Load Stack instance from Experiment. """
        stack = self.experiment.load_stack(self.experiment.stack_ids[0])
        self.assertTrue(isinstance(stack, Stack))
