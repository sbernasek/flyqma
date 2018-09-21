
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



Clones Input
=========

Package scope begins with a set of directories, each containing a single ``.tif`` file. Each of these images should depict a single *Drosophila* eye disc that has been marked with fluorescent reporters, dissected, and imaged.

Images may be regularly-spaced 3D z-stacks, irregularly-spaced 3D collections of layers, or 2D images of individual layers. As segmentation is performed in 2D, overlapping layers must be manually marked for exclusion using the provided selection GUI.

The current implementation is limited to RGB image format. Extension to higher dimensionality is readily possible should additional fluorescence channels become necessary.



Clones Modules
=========

Clones is organized into several modules:

* Data. Components for managing image and cell measurement data. Clones provides three levels of organization:

  1. ``Experiment`` collection of eye discs obtained under similar conditions

  2. ``Stack`` collection of layers depicting an individual eye disc

  3. ``Layer`` 2D RGB cross section of an eye disc

* Measure. Methods for identifying cells and quantifying their expression levels

* Annotation. Methods for automated classification of clonal subpopulations

* Bleedthrough. Methods for correcting fluorescence bleedthrough between reporters

* Selection. Simple matplotlib-based GUI for manual curation of measurement data

* Analysis. Methods for statistical comparison of measurement data



Example Usage
=========

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


Aggregate all measurement data for an experiment

    #!/usr/bin/env python

    data = experiment.aggregate_measurements()


Segment a layer and save the results:

    #!/usr/bin/env python

    layer.segment()

    layer.save()


Perform bleedthrough correction:

  #!/usr/bin/env python

  from clones.bleedthrough.correction import LayerCorrection

  correction = LayerCorrection(layer, niters=60)
  correction.show_correction()
  correction.save()


Launch the selection GUI (in Jupyter):

    #!/usr/bin/env python

    %matplotlib notebook

    stack = experiment.load_stack(stack_ind, full=True)

    gui = GUI.load(stack)



Further Examples
-------------

For detailed usage examples, please refer to the [code](https://github.com/sebastianbernasek/pnt_yan_ratio) used to generate the figures in our manuscript.
