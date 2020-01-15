.. image:: graphics/Northwestern_purple_RGB.png
   :width: 30%
   :align: right
   :alt: nulogo
   :target: https://amaral.northwestern.edu/


.. _filestructure:

File Structure
==============

**Fly-QMA** uses the standardized file structure outlined below. Fly-QMA will adhere to this structure automatically, creating and updating the various subdirectories and files as needed. However, it is also possible to integrate external analysess (such as a segmentation mask) into the Fly-QMA workflow by manually adding them to the standardized file structure.

The standardized file structure is hierarchically organized into three levels:

 1. **EXPERIMENT**: One or more tissue samples imaged under the same conditions.

 2. **STACK**: All images of a particular tissue sample, such as an individual z-stack.

 3. **LAYER**: All analysis relevant to a single 2-D image, such as an individual layer.


Experiments
-----------

Microscopy data should be arranged into a collection of **STACK** directories that reside within an **EXPERIMENT** directory unique to a particular set of experimental conditions. These **STACK** directories are sequentially numbered, beginning with zero.

.. code-block:: bash

   EXPERIMENT
   │
   ├── 0         # First STACK directory
   ├── 1
   └── ... N     # Nth STACK directory


Z-Stacks
--------

Each **STACK** directory contains various components pertinent to all images within the z-stack. These may include:

 - The original ``.tif`` image file depicting a z-stack of an imaginal disc. Images may be regularly-spaced 3D z-stacks or irregularly-spaced 3D collections of one or more layers. If a 2D image is provided, Fly-QMA will assume the z-stack only contains a single layer. *Note that the image file must retain the same name as its parent STACK directory.*
 - A ``metadata.json`` file containing all imaging metadata, e.g. number of layers, number of fluorescence channels, image bit depth, etc.
 - An ``annotation`` subdirectory containing all of the model components used to annotate a particular image stack.
 - A ``layers`` subdirectory containing all of the lower level **LAYER** directories. Layers are sequentially numbered, beginning with zero.

.. code-block:: bash

   EXPERIMENT
   │
   ├── 0
   │   ├── 0.tif
   │   ├── metadata.json
   │   ├── annotation
   │   └── layers
   │        ├── 0        # first LAYER directory
   │        ├── 1
   │        └── ... N    # Nth LAYER directory
   ├── 1
   │
   └── ... N


Layers
------

Each **LAYER** directory contains all components pertinent to an individual 2D layer within the z-stack. These may include:

 - A ``metadata.json`` file containing all layer metadata, such as particular parameter values used.
 - A ``selection`` subdirectory containing a ``selection.npy`` ROI mask. This mask is a 2D numpy array of boolean values in which each element indicates whether a given pixel is within the ROI.  **Users may readily import their own ROI mask by manually replacing this file.** The ``selection`` directory also includes a ``md.json`` file used whose contents are used to indicate whether or not the layer is included within subsequent analyses.
 - A ``correction`` subdirectory containing a parameterized model for performing bleedthrough correction. The ``data.json`` file contains the model parameterization, while ``fit.png`` depicts the model fit and ``correction.png`` shows the resultant correction.
 - A ``segmentation`` subdirectory containing a ``labels.npy`` segmentation mask. This mask is a 2D numpy array of integers in which each element represents a single pixel within the image. The integer value denotes the segment assigned to each pixel, where zero-valued pixels comprise the background. **As this output format is shared by other segmentation platforms (such as skimage), users may readily import their own segmentation by manually replacing this file.** The ``segmentation`` directory may also include an image of the resultant segmentation, stored as ``segmentation.ong``, but this file is not required.
 - A ``measurements`` subdirectory containing two serialized Pandas dataframes. The file ``measurements.hdf`` contains the raw measured pixel intensities for all detected cells or nuclei, while ``processed.hdf`` contains a cached version of the measured data after all analyses (e.g. bleedthrough correction, annotation, etc.) have been applied. The former is used to preserve the original measurements, while the latter is used to cache the results of previous analysis so they may be rapidly retrieved at any time.


.. code-block:: bash

   EXPERIMENT
   │
   ├── 0
   │   ├── 0.tif
   │   ├── metadata.json
   │   ├── annotation
   │   └── layers
   │       ├── 0
   │       │   ├── metadata.json
   │       │   │
   │       │   ├── selection              # ROI selection subdirectory
   │       │   │   ├── md.json
   │       │   │   └── selection.npy
   │       │   │
   │       │   ├── correction             # bleedthrough correction subdirectory
   │       │   │   ├── data.json
   │       │   │   ├── fit.png
   │       │   │   └── correction.png
   │       │   │
   │       │   ├── segmentation
   │       │   │   ├── labels.npy         # segmentation mask (np.ndarray[int])
   │       │   │   └── segmentation.png   # layer image overlayed with segment contours (optional)
   │       │   │
   │       │   └── measurements
   │       │       ├── measurements.hdf   # raw expression measurements
   │       │       └── processed.hdf      # processed expression measurements
   │       │
   │       ├── 1
   │       └── ... N
   ├── 1
   └── ... N


Annotation
----------

In Fly-QMA, annotation entails training a model to identify distinct levels of clonal marker fluorescence, then applying the model within the spatial context of a given image. While annotation is always applied at the **LAYER** level, Fly-QMA supports training the annotation model on each **LAYER** or on the entire **STACK**. The ``annotation`` subdirectory resides at the level used to train the model. Its contents are detailed below. If a model selection procedure is used, all of the trained models are also cached within a ``models`` subdirectory.


.. code-block:: bash

   EXPERIMENT
   │
   ├── 0
   │   ├── 0.tif
   │   ├── metadata.json
   │   ├── layers
   │   └── annotation                     # annotation subdirectory
   │       │
   │       ├── annotation.json            # annotation parameters
   │       │
   │       ├── classifier                 # selected model directory
   │       │   ├── parameters.json        # selected model parameters
   │       │   ├── model.pkl              # pickled mixture model
   │       │   └── values.npy             # data used to fit mixture model
   │       │
   │       └── models                     # model selection directory
   │           ├── parameters.json        # model selection parameters
   │           ├── values.npy             # data used for model selection
   │           ├── classifier_0
   │           ├── classifier_1
   │           └── ... classifier_M       # Mth mixture model directory
   ├── 1
   └── ... N
