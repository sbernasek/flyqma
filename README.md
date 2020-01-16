Fly-QMA
=======

**Fly-QMA** is part of the **NU FlyEye** platform for quantitative analysis of *Drosophila* imaginal discs. The package enables Quantitative Mosaic Analysis (QMA) - that is, it helps users quantify and analyze expression patterns in mosaic tissues.

Expression patterns are typically identified by comparing the fluorescence of target gene reporters between groups of cells. Fly-QMA helps quantify these differences in reporter expression. Quantification entails measuring fluorescence intensities by analyzing images of imaginal discs. These measurements may then used to detect and analyze spatial patterns within each image.

Given confocal microscopy data, Fly-QMA provides:

  - **Segmentation.** Detect cell nuclei within each image

  - **Measurement.** Quantify reporter expression levels

  - **Bleedthrough Control.** Correct for fluorescence bleedthrough

  - **Annotation.** Label distinct subpopulations of cells

  - **ROI Definition.** Manually define regions of interest

  - **Comparison.** Compare expression levels between subpopulations

Please visit the [Fly-QMA homepage](https://sebastianbernasek.github.io/flyqma) for tips on getting started.


Installation
============

Installing Fly-QMA is easy. Set up a working environment running Python 3.6+, then install via  ``pip``:

    pip install flyqma


Getting Started
===============

See the [Fly-QMA tutorial](https://github.com/sebastianbernasek/flyqma/blob/master/tutorial.ipynb).


Examples
========

For additional examples of complete projects utilizing Fly-QMA and the entire **NU FlyEye** platform, check out:

 - [Our Fly-QMA manuscript](https://github.com/sebastianbernasek/flyqma_ms)
 - [Our study](https://github.com/sebastianbernasek/pnt_yan_ratio) of Pnt and Yan expression during retinal patterning in *Drosophila*.
