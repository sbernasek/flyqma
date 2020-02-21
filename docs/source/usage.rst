.. image:: graphics/Northwestern_purple_RGB.png
   :width: 30%
   :align: right
   :alt: nulogo
   :target: https://amaral.northwestern.edu/


Basic Usage
===========

**Fly-QMA** provides a wide range of functionality for measuring and analyzing mosaic imaginal discs. A brief introduction to some basic operations is provided below. For explicit API details please see the :ref:`code documentation <documentation>`.



Organizing Images
-----------------

We recommend first organizing your images in accordance with our recommended hierarchical :ref:`file structure <filestructure>`. In the examples below, ``./data`` would contain several subdirectories, each of which contains a single ``.tif`` file. Once everything is in place, get started by creating an ``Experiment`` instance. This will serve as the entry-point for managing your data in Fly-QMA.

.. code-block:: python

   >>> from flyqma.data import experiments
   >>> path = './data'
   >>> experiment = experiments.Experiment(path)

Lower levels of the data hierarchy may then be accessed in a top-down manner. Methods acting upon lower level instances are executed in place, meaning you won't lose progress by iterating across instances or by coming back to a given instance at a different time.

To select a specific image stack:

.. code-block:: python

   >>> stack_id = 0
   >>> stack = experiment[stack_id]


To select a specific layer:

.. code-block:: python

   >>> layer_id = 0
   >>> layer = stack[layer_id]


To select a specific fluorescence channel:

.. code-block:: python

   >>> channel_id = 0
   >>> channel = layer.get_channel(channel_id)




Segmenting Images
-----------------

See the measurement :ref:`documentation <measurement_docs>` for a list of the specific parameters needed to customize the segmentation routine to suit your data. At a minimum, users must specify the background ``channel`` - that is, the index of the fluorescence channel used to identify cells or nuclei, e.g. a DAPI stain.

To segment an image layer, measure the segment properties, and save the results:

.. code-block:: python

   >>> background_channel = 2
   >>> layer.segment(background_channel)
   >>> layer.save()

Alternatively, to segment all layers within an image stack:

.. code-block:: python

   >>> background_channel = 2
   >>> stack.segment(background_channel, save=True)

In both cases, measurement data are generated on a layer-by-layer basis and are accessed via the ``Layer.data`` attribute. Specifying ``save=True`` or calling ``layer.save()`` ensures that the measurement data will remain available even after the session is terminated.




Analyzing Measurements
----------------------

Measurement data are stored in a `Pandas DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_. Each sample (row) in the DataFrame reflects an individual cell or nuclear fluorescence measurement, whose index denotes a unique Segment ID. Columns depict a mixture of continuous and categorical features pertaining to each sample. By default, these features include:

 - **segment_id:** The unique integer identifier assigned to the segment.
 - **pixel_count:** The total number of pixels within the segment.
 - **centroid_x:** The mean x-coordinate of all pixels within the segment.
 - **centroid_y:** The mean y-coordinate of all pixels within the segment.
 - **chN** - The mean intensity of the Nth channel across all pixels within the segment.
 - **chN_std** - The standard deviation of the Nth channel across all pixels within the segment.
 - **chN_normalized** - The mean intensity of the Nth channel divided by the mean intensity of the background channel.

Additional features are automatically appended to the DataFrame as further analyses are performed. These may include:

 - **chN_predicted:** The estimated contribution of bleedthrough into the measured level of the Nth channel.
 - **chNc:** The bleedthrough-corrected mean intensity of the Nth channel.
 - **chNc_normalized:** The bleedthrough-corrected normalized mean intensity of the Nth channel.
 - **selected:** Boolean flag indicating whether the segment falls within the user-specific ROI.
 - **boundary:**  Boolean flag indicating whether the segment lies within a boundary between differing cell types.
 - **manual_label:** Segment label that was manually assigned using  `FlyEye Silhouette <https://www.silhouette.amaral.northwestern.edu/>`_.

Furthermore, the annotation module may be used to assign one or more labels to each segment. Users are free to specify the names of these additional features as they please.



Aggregating Measurements
------------------------

To aggregate data across all layers in an image stack:

.. code-block:: python

   >>> stack_data = stack.aggregate_measurements()

Similarly, to aggregate measurement data across an entire experiment:

.. code-block:: python

   >>> experiment_data = experiment.aggregate_measurements()

Each of these operations returns cell measurement data in the same DataFrame format discussed above. However, in order to preserve the unique identity of each measurement the index is replaced by a hierarchical index depicting the unique layer and/or stack from which each segment was derived.




.. _gui:

Defining a ROI
--------------

To import an externally generated ROI mask please refer to the :ref:`integration <roi_integration>` section.


Fly-QMA includes a matplot-lib based GUI for selecting a particular region of interest within an image layer. The interface consists of a grid of images in which rows correspond to layers and columns correspond to each of the available fluorescence channels. To launch the GUI for an individual image stack:

.. code-block:: python

   >>> from flyqma.selection.gui import GUI

   >>> # load a stack, including its image
   >>> stack = experiment.load_stack(stack_ind, full=True)

   >>> # create the GUI interface (a still image)
   >>> gui = GUI(stack)

   >>> # connect interface to user input
   >>> gui.connect()


Regions of interest are selected by drawing a selection boundary that encloses them. A selection boundary is defined by a series of sequentially-added points. Click on a layer image to add a point to that layer's selection boundary. Points may be added to any of a layer's fluorescence channels, and will automatically appear withing all other fluorescence channels for that layer. The most recently added point appears as a large red dot, while all other points appear as small yellow dots. Once three or more dots are present in an image layer, the current selection boundary is displayed with a yellow line. Once completed, a given layer might look like:


.. figure:: graphics/example_gui.png
   :scale: 100 %
   :align: center
   :alt: example gui

   **Example:** ROI boundaries for two layers, one of which is excluded.

The GUI offers some basic key commands:

.. code-block:: bash

   T: remove last added point
   Y: remove all points in layer
   W: save ROI selection
   Q: exit GUI

When a selection is saved, a boolean *selected* attribute is added to the layer's cell measurement data indicating whether or not a given cell lies within the layer's selection path. The *selected* attribute may then be used to filter the measurement data during subsequent analysis. The GUI also allows the user to mark entire layers for exclusion using an additional key commands:

.. code-block:: bash

   E: exclude entire layer

Layers marked *excluded* will be masked by a transparent overlay. When these layers are saved, the *selected* attribute is set to False for all of their constituent cell measurements.

A saved GUI may be reopened via the ``GUI.load`` method, at which point further adjustments may be made to each layer.

See the ROI selection :ref:`documentation <selection_docs>` for additional details.




Correcting Bleedthrough
-----------------------

To perform bleedthrough correction:

.. code-block:: python

   >>> from flyqma.bleedthrough.correction import LayerCorrection
   >>> correction = LayerCorrection(layer)
   >>> correction.save()

See the bleedthrough correction :ref:`documentation <bleedthrough_docs>` for additional details and a complete list of available parameters.





Example Projects
----------------

For real usage examples, please refer to the `FlyQMA manuscript <https://doi.org/10.1101/775783>`_ and `our study <https://doi.org/10.1101/430744>`_ of Pnt and Yan expression in the developing eye.
