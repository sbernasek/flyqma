from unittest.mock import patch

from os.path import join, exists
from pandas import DataFrame

from flyqma.data import Stack
from flyqma.data.layers import Layer
from .test_io import TestPaths


class Test02_Stack(TestPaths):
    """
    Tests for Stack instance.
    """

    @classmethod
    def setUpClass(cls):
        """ Initialize test instance of Stack. """
        with patch('builtins.input', side_effect=['yes', 12]):
            stack = Stack(cls.stack_path, bit_depth=cls.bit_depth)
        cls.stack = stack

    @classmethod
    def tearDownClass(cls):
        """ Restore stack directory to original state. """
        cls.stack.restore_directory()

    def test01_paths(self):
        """ Check that child directories are present. """
        tests = [exists(self.stack.layers_path),
                 exists(self.stack.path)]
        self.assertTrue(False not in tests)

    def test02_load_image(self):
        """ Load stack image. """
        self.stack.load_image()
        self.assertTrue(self.stack.stack is not None)

    def test03_load_layer(self):
        """ Load Layer from stack. """
        layer = self.stack.load_layer()
        self.assertTrue(isinstance(layer, Layer))

    def test04_segmentation(self):
        """ Segment all layers in stack.. """
        try:
            self.stack.segment(2)
            success = True
        except:
            success = False
        self.assertTrue(success)

    def test05_train_annotator(self):
        """ Train clone annotation model on entire stack. """
        try:
            self.stack.train_annotator('ch0', save=True, max_num_components=3)
            success = True
        except:
            success = False
        self.assertTrue(success)

    def test06_apply_annotator(self):
        """ Annotate a single layer within the stack. """
        layer = self.stack[0]
        layer.build_graph('ch0')
        layer.annotate()
        self.assertTrue('genotype' in layer.data.columns)

    def test07_aggregate_raw_data(self):
        """ Collect raw measurements from all layers. """
        data = self.stack.aggregate_measurements(raw=True)
        self.assertTrue(isinstance(data, DataFrame))

    def test07_aggregate_data_excluding_boundary(self):
        """ Collect measurements from all layers, excluding boundaries. """
        data = self.stack.aggregate_measurements(exclude_boundary=True)
        self.assertTrue(isinstance(data, DataFrame))
