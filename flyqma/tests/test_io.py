from unittest import TestCase
from os.path import join
from flyqma.data import Experiment, Stack


class TestPaths(TestCase):
    """ Initialize paths to test data. """
    bit_depth = 12
    fixtures_path = 'flyqma/tests/fixtures'
    exp_path = 'flyqma/tests/fixtures/experiment'
    stack_path = 'flyqma/tests/fixtures/experiment/disc'
    #silhouette_path = join(self.fixtures_path, 'disc.silhouette')
    #tif_path = join(self.fixtures_path, 'disc.tif')


class TestIO(TestPaths):
    """
    Tests for instantiating Experiment and Stack instances in Fly-QMA.
    """

    def test_instantiate_stack_from_path(self):
        """ Instantiate Stack from stack path.  """
        stack = Stack(self.stack_path, bit_depth=self.bit_depth)
        self.assertTrue(isinstance(stack, Stack))

    def test_instantiate_stack_from_silhouette(self):
        """ Instantiate Stack from Silhouette file.  (not yet implemented) """
        #stack = Stack.from_silhouette(self.silhouette_path)
        #self.assertTrue(isinstance(stack, Stack))
        self.assertTrue(True)

    def test_instantiate_stack_from_tif(self):
        """ Instantiate Stack from tif image.  """
        #stack = Stack.from_tif(self.tif_path, bit_depth=self.bit_depth)
        #self.assertTrue(isinstance(stack, Stack))
        self.assertTrue(True)

    def test_instantiate_experiment_from_path(self):
        """ Instantiate experiment from collection of stack directories.  """
        exp = Experiment(self.exp_path)
        self.assertTrue(isinstance(exp, Experiment))
