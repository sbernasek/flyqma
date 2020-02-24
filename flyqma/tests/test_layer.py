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

    annotated_channel = 'ch0'
    label_name = 'TEST'

    @classmethod
    def setUpClass(cls):
        """ Initialize test instance of Stack. """
        with patch('builtins.input', side_effect=['yes', 12]):
            stack = Stack(cls.stack_path, bit_depth=cls.bit_depth)
        stack.load_image()
        cls.stack = stack
        cls.layer = cls.stack[0]

    @classmethod
    def tearDownClass(cls):
        """ Restore stack directory to original state. """
        cls.stack.restore_directory()

    def test00_import_segmentation_mask(self):
        """ Import a segmentation mask and generate measurements. """
        self.layer.import_segmentation_mask(self.segmentation_mask_path, 2)
        self.assertTrue(self.layer.is_segmented)

    def test01_import_roi_mask(self):
        """ Define region of interest by importing an ROI mask. """
        self.layer.import_roi_mask(self.roi_mask_path, save=True)
        self.assertTrue('selected' in self.layer.data.columns)

    def test02_build_graph(self):
        """ Build cell adjacency graph. """
        self.layer.build_graph(self.annotated_channel)
        self.assertTrue(self.layer.graph is not None)

    def test03_train_annotator(self):
        """ Train clone annotation model on individual layer. """
        self.layer.train_annotator(self.annotated_channel)
        self.assertTrue(self.layer.annotator is not None)

    def test04_apply_annotator(self):
        """ Annotate layer. """
        self.layer.apply_annotation(label=self.label_name)
        self.assertTrue(self.label_name in self.layer.data.columns)

    def test05_mark_label_boundaries(self):
        """ Mark boundaries between regions with different labels. """
        self.layer.mark_boundaries(self.label_name, max_edges=1)
        self.assertTrue('boundary' in self.layer.data.columns)

    def test06_mark_label_regions(self):
        """ Mark regions in which each label is found. """
        self.layer.apply_concurrency(basis=self.label_name)
        label = list(self.layer.data[self.label_name].unique())[0]
        key = 'concurrent_{:d}'.format(label)
        self.assertTrue(key in self.layer.data.columns)

    def test07_build_attribute_mask(self):
        """ Build attribute mask for layer. """
        mask = self.layer.build_attribute_mask(self.label_name)
        self.assertTrue(type(mask) == np.ma.core.MaskedArray)

    def test08_show_annotation(self):
        """ Show layer annotation. """
        if self.layer.include:
            channel, label = self.annotated_channel, self.label_name
            try:
                fig = self.layer.show_annotation(channel, label)
                success = True
            except:
                success = False
            self.assertTrue(success)

    def test09_select_channel(self):
        """ Select individual fluorescence channel. """
        channel = self.layer.get_channel(1)
        self.assertTrue(type(channel) == ImageScalar)

    def test10_bleedthrough_correction(self):
        """ Correct measurements for fluorescence bleedthrough. """
        correction = LayerCorrection(self.layer, xvar=0, yvar=1)
        correction.save()
        self.assertTrue('ch1c' in self.layer.data.columns)
