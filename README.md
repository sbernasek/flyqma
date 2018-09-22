
Clones Overview
===========

Clones is a platform for analyzing protein expression patterns in the developing Drosophila eye. The package is particularly focused on comparing expression between distinct clonal subpopulations within the eye epithelium.

Expression patterns are typically identified by comparing the fluorescence of target gene reporters between groups of cells. Clones helps quantify these differences in reporter expression. Quantification entails measuring fluorescence intensities by analyzing experimentally collected images of fixed eye discs. Such measurements may then used to identify and compare distinct subpopulations within the eye field.

Given confocal microscopy data, clones facilitates:

  - **Segmentation:** detection of cell nuclei within each image

  - **Measurement:** quantification of reporter expression levels

  - **Bleedthrough Control:** correction of fluorescence bleedthrough

  - **Annotation:** identification of distinct subpopulations of cells

  - **Cell Selection:** manual selection of cells for comparison

  - **Comparison:** statistical comparison of subpopulations


The initial release is primarily limited to basic tools required to replicate [our study](https://github.com/sebastianbernasek/pnt_yan_ratio) of Pnt and Yan expression during retinal patterning in *Drosophila*. This functionality may be expanded in later releases.



Installation
=========

After downloading the [latest distribution](https://github.com/sebastianbernasek/clones/archive/v0.1.0-beta.tar.gz), the simplest method is to install via ``pip``:

    pip install clones-0.1.0-beta.tar.gz



Clones Package Structure
=========

Clones is organized into several submodules:

* ``clones.data`` provides components for managing image and cell measurement data across three levels of organization:

  1. ``Layer`` instances contain a 2D RGB cross section of an eye disc

  2. ``Stack`` instances contain a collection of layers obtained from the same eye disc.

  3. ``Experiment`` instances contain a collection of eye discs obtained under similar conditions.

* ``clones.measure`` provides methods for identifying cells and quantifying their expression levels.

* ``clones.spatial`` provides methods for analyzing the spatial arrangement of identified cells.

* ``clones.annotation`` provides methods for automated classification of clonal subpopulations.

* ``clones.bleedthrough`` provides methods for correcting fluorescence bleedthrough between reporters.

* ``clones.selection`` provides a matplotlib-based GUI for spatial curation of measurement data.

* ``clones.selection`` provides methods for statistical comparison of measurement data.


Clones Usage
=========


Input Data
-------------

While the methods provided by ``clones`` are readily applicable to individual layers or image stacks, we recommend starting with a standardized input file structure. Microscopy data should be arranged into a collection of sequentially numbered "stack directories" that reside within a directory unique to a particular set of experimental conditions:

    /experiment
    │
    ├── 0         # First stack directory
    ├── 1
    └── ... N     # Nth stack directory

Each stack directory should contain a single ``.tif`` file depicting a *Drosophila* eye disc that has been marked with fluorescent reporters, dissected, and imaged:

    /experiment
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


Getting Started
-------------

All measurements and analyses are performed in place. This means that new subdirectories and files are added to a stack directory each time a new segmentation, measurement, annotation, bleedthrough correction, or cell selection is saved. Saving one of these operations will overwrite any existing files of the same type.

To begin using clones, create an ``Experiment`` instance by passing the ``/experiment`` path to the object constructor:

    experiment = Experiment(path='/experiment')

This instance will serve as a central hub for measuring and analyzing all of the stacks in the ``/experiment`` directory. To access an individual stack:

    # load specific stack
    stack = experiment.load_stack(stack_id)

    # alternatively, by sequential iteration
    for stack in experiment:
      stack.do_stuff()

The ``experiment.load_stack()`` method includes a ``full`` keyword argument that may be set to False in order to skip loading the stack's ``.tif`` file into memory. This offers some performance benefit when only saved measurement data are needed. Of course, loading the image data is necessary if any segmentation, measurement, cell selectiom, or bleedthrough correction operations are to be performed.

To begin analyzing an image stack, layers must be added to the corresponding stack directory. The ``Stack.initialize()`` method creates a ``layers`` subdirectory containing an additional subdirectory for each layer in the 3D image stack. A stack metadata file is similarly added to the stack directory at this time, resulting in:

    /experiment
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

    # load specific layer
    layer = stack.load_layer(layer_id)

    # alternatively, by sequential iteration
    for layer in stack:
      layer.do_stuff()


Expression Measurement
-------------

For a given layer, segmentation and expression quantification are performed by calling the ``layer.segment`` method.
See the ``layer.segment`` documentation for an overview of customizable image preprocessing, seed detection, or segmentation parameters. Upon completion, the results may be saved by calling ``layer.save()``. This saves the segmentation parameters within a layer metadata file and creates a ``segmentation`` subdirectory containing a segment labels mask and the corresponding raw expression measurement data:

    /experiment
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
-------------

The data stored in the ``layer.measurements`` attribute and ``measurements.hdf`` file reflect raw measurements of mean pixel fluorecence intensity for each identified cell contour. These measurements may then be subject to one or more processing operations such as:

  * Annotation: automated assignment of cell types to each contour
  * Bleedthrough correction: correction for bleedthrough effects between fluorescence channels
  * Cell selection: manual curation of layers or regions of layers to be included in the dataset, e.g. exclusion of overlapping layers

The objects that perform these operations all behave in a similar manner. They are manually defined for each disc (see Jupyter notebooks for examples), but may then be saved for repeated use. When saved, each object creates its own subdirectory within the corresponding layer directory:

    /experiment
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

    /experiment
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


Example Usage
-------------

Load an experiment, stack, or layer from a directory containing subdirectories of ``.tif`` files:

    #!/usr/bin/env python

    from clones.data import experiments

    # load a specific experiment
    path = './data/clones/yanE833'
    experiment = experiments.Experiment(path)

    # load a specific stack
    stack_id = 0
    stack = experiment[stack_id]

    # load a specific layer
    layer_id = 0
    layer = stack[layer_id]


Segment an image layer and save the results:

    #!/usr/bin/env python

    layer.segment()

    layer.save()


Perform bleedthrough correction:

  #!/usr/bin/env python

  from clones.bleedthrough.correction import LayerCorrection

  correction = LayerCorrection(layer, niters=60)
  correction.show_correction()
  correction.save()


Launch the cell selection GUI for a fully loaded image stack:

    #!/usr/bin/env python

    from clones.selection.gui import GUI

    stack = experiment.load_stack(stack_ind, full=True)

    gui = GUI.load(stack)


Aggregate all measurement data for an experiment

    #!/usr/bin/env python

    data = experiment.aggregate_measurements()


Further Examples
-------------

For detailed usage examples, please refer to the [code](https://github.com/sebastianbernasek/pnt_yan_ratio) used to generate the figures in our manuscript.
