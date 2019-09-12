.. image:: graphics/Northwestern_purple_RGB.png
   :width: 30%
   :align: right
   :alt: nulogo
   :target: https://amaral.northwestern.edu/


.. _start:

Getting Started
===============

The fastest way to familiarize yourself with **Fly-QMA** is to start with a working example. Feel free to use the data from `our study <https://github.com/sebastianbernasek/pnt_yan_ratio>`_ of Pnt and Yan expression during eye development.

We recommend reading the sections below before working with your own microscopy data.

.. Note::
   The initial release of the **NU FlyEye** platform only supports three-channel RGB microscopy data. The available reporter colors are thus nominally limited to red, green, and blue. One of these reporter colors must be reserved for a nuclear marker in order to facilitate segmentation. This leaves at most two reporter colors available for measuring target gene expression in any one experiment.


Data Format
-----------

**Fly-QMA** requires one or more 2-D cross-sectional images of each eye disc. These images may be supplied in 3-D ``.tif`` format, but individual layers must be spaced far enough apart to avoid capturing the same cells twice. Regulary spacing between layers is not required. To analyze full 3-D RGB image stacks, please see our accompanying package **FlyEye Analysis**.


.. warning::
   **Fly-QMA** prioritizes high-throughput data collection by circumventing the time-intensive process of manually labeling individual cell contours. Because cells often span several adjacent layers in a confocal z-stack, and image segmentation is performed on a layer-by-layer basis, users must supply non-overlapping cross-sectional images of the eye field in order to avoid making duplicate measurements of the same cell. Alternatively, overlapping layers may be manually excluded using the included :ref:`cell selection tool <gui>`.


Data Management
---------------

We recommend a standardized input file structure. Microscopy data should be arranged into a collection of sequentially numbered "stack directories" that reside within a directory unique to a particular set of experimental conditions:

.. code-block:: bash

   experiment
   │
   ├── 0         # First stack directory
   ├── 1
   └── ... N     # Nth stack directory

Each stack directory should contain a single ``.tif`` file depicting a *Drosophila* eye disc that has been marked with fluorescent reporters, dissected, and imaged:

.. code-block:: bash

   experiment
   │
   ├── 0
   │   └── 0.tif   # 3D RGB image
   │
   ├── 1
   │   └── 1.tif
   │
   └── ... N
           └── N.tif

Images may be regularly-spaced 3D z-stacks or irregularly-spaced 3D collections of one or more layers. The current implementation is limited to RGB color format. Extension to higher dimensionality would not be difficult should additional fluorescence channels become necessary.


Loading Data
------------

All measurements and analyses are performed in place. This means that new subdirectories and files are added to a stack directory each time a new segmentation, measurement, annotation, bleedthrough correction, or cell selection is saved. Saving one of these operations will overwrite any existing files of the same type.

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

The ``experiment.load_stack()`` method includes a ``full`` keyword argument that may be set to False in order to skip loading the stack's ``.tif`` file into memory. This offers some performance benefit when only saved measurement data are needed. Of course, loading the image data is necessary if any segmentation, measurement, cell selectiom, or bleedthrough correction operations are to be performed.

To begin analyzing an image stack, layers must be added to the corresponding stack directory. The ``Stack.initialize()`` method creates a ``layers`` subdirectory containing an additional subdirectory for each layer in the 3D image stack. A stack metadata file is similarly added to the stack directory at this time, resulting in:

.. code-block:: bash

   experiment
   │
   ├── 0                   # First stack directory (individual eye disc)
   │   ├── 0.tif           # 3D RGB image
   │   ├── metadata.json   # stack metadata (number of layers, image bit depth, etc.)
   │   └── layers
   │       ├── 0           # first layer
   │       ├── 1
   │       └── ... M       # Mth layer
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
See the ``layer.segment`` documentation for an overview of customizable image preprocessing, seed detection, or segmentation parameters. Upon completion, the results may be saved by calling ``layer.save()``. This saves the segmentation parameters within a layer metadata file and creates a ``segmentation`` subdirectory containing a segment labels mask and the corresponding raw expression measurement data:


.. code-block:: bash

   experiment
   │
   ├── 0                   # First stack directory (individual eye disc)
   │   ├── 0.tif           # 3D RGB image
   │   ├── metadata.json   # stack metadata (number of layers, image bit depth, etc.)
   │   └── layers
   │       ├── 0
   │       │   ├── metadata.json          # layer metadata (background channel, parameter values, etc.)
   │       │   └── segmentation
   │       │       ├── labels.npy         # segment labels mask (np.ndarray[int])
   │       │       ├── measurements.hdf   # raw expression measurements
   │       │       └── segmentation.png   # layer image overlayed with segment contours (optional)
   │       ├── 1
   │       └── ... M
   ├── 1
   └── ... N


Data Processing
---------------

The data stored in the ``layer.measurements`` attribute and ``measurements.hdf`` file reflect raw measurements of mean pixel fluorecence intensity for each identified cell contour. These measurements may then be subject to one or more processing operations such as:

  * Annotation: automated assignment of cell types to each contour
  * Bleedthrough correction: correction for fluorescence bleedthrough between reporters
  * Cell selection: manual exclusion of layers or regions of layers from the dataset

The objects that perform these operations all behave in a similar manner. They are manually defined for each disc (see Jupyter notebooks for examples), but may then be saved for repeated use. When saved, each object creates its own subdirectory within the corresponding layer directory:

.. code-block:: bash

    experiment
    │
    ├── 0
    │   ├── 0.tif
    │   ├── metadata.json
    │   └── layers
    │       ├── 0
    │       │   ├── metadata.json
    │       │   ├── segmentation
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

The added subdirectories include all the files and metadata necessary to load and execute the data processing operations performed by the respective object. Saved operations are automatically applied to the raw measurement data each time a layer is loaded. Under this setup, only raw measurements are ever stored long term. Processed measurements are only accessible via the ``layer.data`` attribute when a layer exists in local memory. They may be aggregated across layers via ``stack.aggregate_measurements()`` and across all stacks in an experiment via ``experiment.aggregate_measurements()``.

Note that cell-based classifiers are fit to an entire stack, rather than to each of its individual layers. Consequently, these classifiers are stored in their own subdirectory below the stack header:


.. code-block:: bash

   experiment
   │
   ├── 0
   │   ├── 0.tif
   │   ├── metadata.json
   │   ├── layers
   │   └── cell_classifier
   │       ├── parameters.json
   │       ├── values.npy
   │       └── classifier.pdf
   ├── 1
   └── ... N
