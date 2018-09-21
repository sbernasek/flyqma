
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

* Data. Components for managing **FlyEye Silhouette** data. FlyEye provides three levels of organization:

  1. ``Cells`` objects contain one or more expression level measurements

  2. ``Disc`` objects contain all expression level measurements from a single ``.silhouette`` file

  3. ``Experiment`` objects contain multiple ``Disc`` instances collected under similar conditions

* Processing. Methods for converting cell measurements into expression time series.

* Dynamics. Methods for time series visualization.

* Analysis. Methods for quantitative analysis of expression data.



Example Usage
=========

Import an experiment from a directory containing ``.silhouette`` files:

    #!/usr/bin/env python

    from flyeye.data import experiments

    path = './silhouette_data'
    experiment = experiments.Experiment(path)


Select a specific disc:

    disc_id = 2
    disc = experiment.discs[disc_id]


Select a specific cell type:

    cell_type = 'pre'
    cells = disc.select_cell_type(cell_type)


Plot expression dynamics:

    fluorescence_channel = 'green'
    cells.plot_dynamics(fluorescence_channel)


Further Examples
-------------

For detailed usage examples, please refer to the [code](https://github.com/sebastianbernasek/pnt_yan_ratio) used to generate the figures in our manuscript.
