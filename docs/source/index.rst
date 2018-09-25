.. image:: graphics/Northwestern_purple_RGB.png
   :width: 30%
   :align: right
   :alt: nulogo

=================
NU FlyEye: Clones
=================

.. image:: graphics/header.png
   :width: 100%
   :align: center
   :alt: header

**NU FlyEye: Clones** is part of the **NU FlyEye** platform for studying gene expression in the developing *Drosophila* eye. The clones package enables quantitative comparisons of expression between distinct clonal subpopulations within the eye epithelium.

Expression patterns are typically identified by comparing the fluorescence of target gene reporters between groups of cells. **NU FlyEye: Clones** helps quantify these differences in reporter expression. Quantification entails measuring fluorescence intensities by analyzing experimentally collected images of fixed eye discs. Measurements are then used to identify and compare distinct subpopulations within the eye field.

Given confocal microscopy data, the clones package facilitates:

  - **Segmentation.** Detect cell nuclei within a microscope image

  - **Measurement.** Quantify reporter expression levels

  - **Bleedthrough Control.** Correct for fluorescence bleedthrough

  - **Annotation.** Identify distinct subpopulations of cells

  - **Cell Selection.** Manually select cells for comparison

  - **Comparison.** Compare expression levels between subpopulations


The initial release is primarily limited to basic tools required to replicate `our study <https://github.com/sebastianbernasek/pnt_yan_ratio>`_ of Pnt and Yan expression during retinal patterning in *Drosophila*. We intend to expand this functionality in later releases.


.. toctree::
   :hidden:
   :maxdepth: 2

   INSTALLATION <installation>
   DOCUMENTATION <documentation>
   GETTING STARTED <start>
   EXAMPLE USAGE <usage>
   CONTACT US <contact>
