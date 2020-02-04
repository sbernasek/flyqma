from unittest.mock import patch

from os.path import join, exists
import numpy as np

from ..data import Stack
from ..data.images import ImageScalar
from ..bleedthrough.correction import LayerCorrection

from .test_io import TestPaths


class Test03_Layer(TestPaths):
    """
    Tests for Layer instance.
    """

    label_name = 'TEST'

    @classmethod
    def setUpClass(cls):
        """ Initialize test instance of Stack. """
        with patch('builtins.input', side_effect=['yes', 12]):
            stack = Stack(cls.stack_path, bit_depth=cls.bit_depth)
        stack.load_image()
        stack.segment(2)
        cls.stack = stack
        cls.layer = cls.stack[0]

    @classmethod
    def tearDownClass(cls):
        """ Restore stack directory to original state. """
        cls.stack.restore_directory()

    def test00_define_roi(self):
        """ Define region of interest. """
        self.layer.define_roi(self.layer.data, self.roi_mask_path)
        self.assertTrue('selected' in self.layer.data.columns)

    def test01_build_graph(self):
        """ Build cell adjacency graph. """
        self.layer.build_graph('ch0')
        self.assertTrue(self.layer.graph is not None)

    def test02_train_annotator(self):
        """ Train clone annotation model on individual layer. """
        self.layer.train_annotator('ch0')
        self.assertTrue(self.layer.annotator is not None)

    def test03_apply_annotator(self):
        """ Annotate layer. """
        self.layer.apply_annotation(label=self.label_name)
        self.assertTrue(self.label_name in self.layer.data.columns)

    def test04_mark_label_boundaries(self):
        """ Mark boundaries between regions with different labels. """
        self.layer.mark_boundaries(self.label_name, max_edges=1)
        self.assertTrue('boundary' in self.layer.data.columns)

    def test05_mark_label_regions(self):
        """ Mark regions in which each label is found. """
        self.layer.apply_concurrency(basis=self.label_name)
        label = list(self.layer.data[self.label_name].unique())[0]
        key = 'concurrent_{:d}'.format(label)
        self.assertTrue(key in self.layer.data.columns)

    def test06_build_attribute_mask(self):
        """ Build attribute mask for layer. """
        mask = self.layer.build_attribute_mask(self.label_name)
        self.assertTrue(type(mask) == np.ma.core.MaskedArray)

    def test07_select_channel(self):
        """ Select individual fluorescence channel. """
        channel = self.layer.get_channel(1)
        self.assertTrue(type(channel) == ImageScalar)

    def test08_bleedthrough_correction(self):
        """ Correct measurements for fluorescence bleedthrough. """
        correction = LayerCorrection(self.layer, xvar=0, yvar=1)
        correction.save()
        self.assertTrue('ch1c' in self.layer.data.columns)
