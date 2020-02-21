.. image:: graphics/Northwestern_purple_RGB.png
   :width: 30%
   :align: right
   :alt: nulogo
   :target: https://amaral.northwestern.edu/


.. _integration:

External Integrations
=====================

Fly-QMA supports integration with external analysis platforms. See the subsections below for detailed instructions.


Segmentation Masks
------------------

Image segmentation can often be strongly context-dependent, limiting the utility of once-size-fits-all strategies. Fly-QMA was therefore designed to support integration with external segmentation tools by allowing users to seemlessly import externally generated segmentation masks. Segmentation masks are two-dimensional ``.npy`` arrays whose dimensions match those of the image to be segmented. Each array element is an integer value that denotes the label assigned to the corresponding pixel in the segmented image. Zero-valued pixels denote the background. This format is shared by several common segmentation platforms, including `scikit-image <https://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html>`_.

Segmentation masks are imported by calling the ``Layer.import_segmentation_mask`` method. Expression measurements will then automatically be generated using the provided mask. The ``save=True`` argument tells Fly-QMA to save a copy of the mask to the appropriate **LAYER** subdirectory in order to ensure that the mask remains available even after the Layer instance is destroyed.

.. code-block:: python

   >>> segmentation_mask_path = './segmentation.npy'
   >>> channel = 2
   >>> layer.import_segmentation_mask(segmentation_mask_path, channel, save=True)


.. _roi_integration:

ROI Masks
---------

Fly-QMA also supports importing an externally generated ROI mask. Like segmentation masks, ROI masks are two-dimensional ``.npy`` arrays whose dimensions match those of the corresponding microscope image. Elements in the ROI mask are binary, taking on either integer values of 0/1 or boolean values of True/False. Values of 1 or True denote that the corresponding pixel is within the region of interest. Please note that the ROI must be a single contiguous region. Support for multiple disconnected regions of interest within a single image is currently under development.

ROI masks are imported by calling the ``Layer.import_roi_mask`` method. A boolean ``selected`` attribute will then automatically be added to the corresponding measurement data, in which a value of True denotes that a given measurement falls within the defined region of interest. The ``save=True`` argument tells Fly-QMA to save a copy of the mask to the appropriate LAYER subdirectory in order to ensure that the ROI remains available even after the Layer instance is destroyed.

.. code-block:: python

   >>> roi_mask_path = './roi_mask.npy'
   >>> layer.import_roi_mask(roi_mask_path, save=True)
