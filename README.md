
Clones Overview
===========

Clones is a platform for analyzing protein expression patterns in the developing Drosophila eye. The package is particularly focused on comparing expression between distinct clonal subpopulations within the eye epithelium.

Expression patterns are typically identified by comparing the fluorescence of target gene reporters between groups of cells. Clones helps quantify these differences in reporter expression. Quantification entails measuring fluorescence intensities by analyzing experimentally collected images of fixed eye discs. Such measurements may then used to identify and compare distinct subpopulations within the eye field.

The initial release is primarily limited to basic tools required to replicate [our study](https://github.com/sebastianbernasek/pnt_yan_ratio) of Pnt and Yan expression during retinal patterning in *Drosophila*. Given confocal microscopy data, the present version of ``clones`` facilitates:

  - **Segmentation:** detection of cell nuclei within each image

  - **Measurement:** quantification of reporter expression levels

  - **Bleedthrough Control:** correction of fluorescence bleedthrough

  - **Annotation:** identification of distinct subpopulations of cells

  - **Cell Selection:** manual selection of cells for comparison

  - **Comparison:** statistical comparison of subpopulations


Please refer to the [documentation](https://sebastianbernasek.github.io/clones/index.html#) for a complete description of all ``clones`` submodules and functions.


Installation
=========

After downloading the [latest distribution](https://github.com/sebastianbernasek/clones/archive/v0.1-beta.tar.gz), the simplest method is to install via ``pip``:

    pip install clones-0.1.0-beta.tar.gz


Example Usage
-------------

Please visit the [clones documentation](https://sebastianbernasek.github.io/clones/index.html#) for tips on getting started with analyzing your data.

For an example project making extensive use of ``clones``, check out [our study](https://github.com/sebastianbernasek/pnt_yan_ratio) of Pnt and Yan expression during retinal patterning of the *Drosophila* eye.
