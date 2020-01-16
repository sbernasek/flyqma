.. image:: graphics/Northwestern_purple_RGB.png
   :width: 30%
   :align: right
   :alt: nulogo
   :target: https://amaral.northwestern.edu/


.. _start:

Getting Started
===============

The fastest way to familiarize yourself with **Fly-QMA** is to start with a working example. We recommend starting with the Fly-QMA `Tutorial <https://github.com/sebastianbernasek/flyqma/blob/master/tutorial.ipynb>`_.

Before working with your own microscopy data, we also suggest reading the sections below.


Image Preparation
-----------------

**Fly-QMA** uses a hierarchical :ref:`file structure <filestructure>` that is organized into three levels:

 1. **EXPERIMENT**: One or more tissue samples imaged under the same conditions.

 2. **STACK**: All images of a particular tissue sample, such as an individual z-stack.

 3. **LAYER**: All analysis relevant to a single 2-D image, such as an individual layer.

Microscopy data should be arranged into a collection of sequentially numbered and zero-indexed **STACK** directories that reside within a particular **EXPERIMENT** directory:

.. code-block:: bash

   EXPERIMENT
   │
   ├── 0         # First STACK
   ├── 1
   └── ... N     # Nth STACK

Each **STACK** directory should contain one or more 2-D images of a unique tissue sample. Images must be supplied in ``.tif`` format with ZXYC orientation, and must share the same name as the parent **STACK** directory, e.g. EXPERIMENT/1/1.tif. Each image file may depict a single layer, a regularly-spaced z-stack, or an irregularly-spaced collections of layers.

.. code-block:: bash

   EXPERIMENT
   │
   ├── 0
   │   └── 0.tif   # 3-D image with ZXYC orientation
   │
   ├── 1
   │   └── 1.tif
   │
   └── ... N
           └── N.tif


.. warning::
   Image segmentation is performed on a layer-by-layer basis. Because cells often span several adjacent layers in a confocal z-stack, individual layers must be spaced far enough apart to avoid measuring the same cells twice. Overlapping layers may also be manually excluded using the provided :ref:`ROI Selector <selection_docs>`.


Loading Images
--------------

All measurements and analyses are performed in place. This means that new subdirectories and files are added to a stack directory each time a new segmentation, measurement, annotation, bleedthrough correction, or region of interest selection is saved. Saving one of these operations will overwrite any existing files of the same type.

To begin using **Fly-QMA**, create an ``Experiment`` instance by passing the ``/experiment`` path to the object constructor:

.. code-block:: python

    experiment = Experiment(path='/experiment')

This instance will serve as a central hub for measuring and analyzing all of the stacks in the ``/experiment`` directory. To access an individual stack:

.. code-block:: python

    # load specific stack
    stack = experiment.load_stack(stack_id)

    # alternatively, by sequential iteration
    for stack in experiment:
      stack.do_stuff()

The ``experiment.load_stack()`` method includes a ``full`` keyword argument that may be set to False in order to skip loading the stack's ``.tif`` file into memory. This offers some performance benefit when only saved measurement data are needed. Of course, loading the image data is necessary if any segmentation, measurement, region of interest selection, or bleedthrough correction operations are to be performed.

To begin analyzing an image stack, layers must be added to the corresponding stack directory. The ``Stack.initialize()`` method creates a ``layers`` subdirectory containing an additional subdirectory for each **LAYER** in the 3D image stack. A stack metadata file is similarly added to the **STACK** directory at this time, resulting in:

.. code-block:: bash

   EXPERIMENT
   │
   ├── 0
   │   ├── 0.tif
   │   ├── metadata.json   # stack metadata (number of layers, image bit depth, etc.)
   │   └── layers
   │       ├── 0           # first LAYER directory
   │       ├── 1
   │       └── ... M       # Mth LAYER directory
   │
   ├── 1
   └── ... N

Image layers may now be analyzed individually. To access an individual layer:

.. code-block:: python

    # load specific layer
    layer = stack.load_layer(layer_id)

    # alternatively, by sequential iteration
    for layer in stack:
      layer.do_stuff()


Measuring Expression
--------------------

For a given layer, segmentation and expression quantification are performed by calling the ``layer.segment`` method.
See the ``flyqma.measurement`` :ref:`documentation <measurement_docs>` for an overview of customizable image preprocessing, seed detection, or segmentation parameters. Measurements for each contour are generated automatically.

Upon completion, the segmentation results and corresponding measurements may be saved by calling ``layer.save()``. This saves the segmentation parameters within a layer metadata file and creates a ``segmentation`` subdirectory containing a segment labels mask. It also creates a ``measurements`` subdirectory containing the corresponding raw expression measurement data as well as a copy subject to all subsequent processing operations. The raw measurements will remain the same until a new segmentation is executed and saved, while the processed measurements are updated each time a new operation is applied and saved.


.. code-block:: bash

   EXPERIMENT
   │
   ├── 0
   │   ├── 0.tif
   │   ├── metadata.json
   │   └── layers
   │       ├── 0
   │       │   ├── metadata.json          # layer metadata (background channel, parameter values, etc.)
   │       │   ├── segmentation
   │       │   │   ├── labels.npy         # segmentation mask (np.ndarray[int])
   │       │   │   └── segmentation.png   # layer image overlayed with segment contours (optional)
   │       │   └── measurements
   │       │       ├── measurements.hdf   # raw expression measurements
   │       │       └── processed.hdf      # processed expression measurements
   │       ├── 1
   │       └── ... M
   ├── 1
   └── ... N


Analysis
--------

The data stored in the ``layer.measurements`` attribute and ``measurements.hdf`` file reflect raw measurements of mean pixel fluorecence intensity for each identified cell contour. These measurements may then be subject to one or more processing operations such as:

  * Annotation: automated assignment of cell types to each contour
  * Bleedthrough correction: correction for fluorescence bleedthrough between reporters
  * Region of interest selection: manual exclusion of layers or regions of layers from the dataset

The objects that perform these operations all behave in a similar manner. They are manually defined for each disc (see the Tutorial), but may then be saved for repeated use. When saved, each object creates its own subdirectory within the corresponding layer directory:

.. code-block:: bash

    EXPERIMENT
    │
    ├── 0
    │   ├── 0.tif
    │   ├── metadata.json
    │   └── layers
    │       ├── 0
    │       │   ├── metadata.json
    │       │   ├── segmentation
    │       │   │   └── ...
    │       │   ├── measurements
    │       │   │   └── ...
    │       │   ├── annotation
    │       │   │   └── ...
    │       │   ├── correction
    │       │   │   └── ...
    │       │   └── selection
    │       │       └── ...
    │       ├── 1
    │       └── ... M
    ├── 1
    └── ... N

The added subdirectories include all the files and metadata necessary to load and execute the data processing operations performed by the respective object. Saved operations are automatically applied to the raw measurement data each time a layer is loaded. Processed measurements are accessible via the ``layer.data`` attribute when a layer exists in local memory. They may also be aggregated across layers via ``stack.aggregate_measurements()`` and across all stacks in an experiment via ``experiment.aggregate_measurements()``.

Note that annotation models may also be fit to an entire stack, rather than to each of its individual layers. Consequently, these annotation models are stored in their own ``annotation`` subdirectory below the stack header. If a model selection procedure is used, all of the trained models are similarly saved within a ``models`` subdirectory.


.. code-block:: bash

   EXPERIMENT
   │
   ├── 0
   │   ├── 0.tif
   │   ├── metadata.json
   │   ├── layers
   │   └── annotation                     # stack annotator directory
   │       │
   │       ├── annotation.json            # annotation parameters
   │       │
   │       ├── classifier                 # selected model directory
   │       │   ├── parameters.json        # selected model parameters
   │       │   ├── model.pkl              # pickled mixture model
   │       │   └── values.npy             # samples used to fit mixture model
   │       │
   │       └── models                     # model selection directory
   │           ├── parameters.json        # model selection parameters
   │           ├── values.npy             # values used for model selection
   │           ├── classifier_0
   │           ├── classifier_1
   │           └── ... classifier_M       # Mth mixture model directory
   ├── 1
   └── ... N
