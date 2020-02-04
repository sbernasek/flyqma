from unittest import TestCase
from unittest.mock import patch

from os.path import join
from shutil import move, rmtree
from flyqma.data import Experiment, Stack


class TestPaths(TestCase):
    """ Initialize paths to test data. """
    bit_depth = 12
    fixtures_path = 'flyqma/tests/fixtures'
    experiment_path = join(fixtures_path, 'experiment')
    stack_path = join(experiment_path, 'disc')
    #silhouette_path = join(self.fixtures_path, 'disc.silhouette')
    tif_path = join(fixtures_path, 'disc.tif')
    roi_mask_path = join(fixtures_path, 'roi_mask.npy')
    segmentation_mask_path = join(fixtures_path, 'segmentation_mask.npy')


class Test01_Instantiation(TestPaths):
    """
    Tests for instantiating Experiment and Stack instances in Fly-QMA.
    """

    def test00_instantiate_stack_from_path(self):
        """ Instantiate Stack from stack path.  """
        with patch('builtins.input', side_effect=['yes', 12]):
            stack = Stack(self.stack_path, bit_depth=self.bit_depth)
        self.assertTrue(isinstance(stack, Stack))

    #def test01_instantiate_stack_from_silhouette(self):
        """ Instantiate Stack from Silhouette file.  (not yet implemented) """
        #stack = Stack.from_silhouette(self.silhouette_path)
        #self.assertTrue(isinstance(stack, Stack))
        #self.assertTrue(True)

    def test02_instantiate_experiment_from_path(self):
        """ Instantiate experiment from collection of stack directories.  """
        with patch('builtins.input', side_effect=['yes', 12]):
            experiment = Experiment(self.experiment_path)
        self.assertTrue(isinstance(experiment, Experiment))

    def test03_instantiate_stack_from_tif(self):
        """ Instantiate Stack from tif image.  """
        with patch('builtins.input', side_effect=['yes', 12]):
            stack = Stack.from_tif(self.tif_path, bit_depth=self.bit_depth)
        self.assertTrue(isinstance(stack, Stack))

        # restore original tif and remove stack directory
        move(stack.tif_path, join(stack.path, '..'))
        stack.restore_directory()
        rmtree(stack.path)
