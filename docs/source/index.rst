.. image:: graphics/Northwestern_purple_RGB.png
   :width: 30%
   :align: right
   :alt: nulogo
   :target: https://amaral.northwestern.edu/


Fly-QMA
=======


**Fly-QMA** is part of the **NU FlyEye** platform for quantitative analysis of *Drosophila* imaginal discs. The package enables Quantitative Mosaic Analysis (QMA) - that is, it helps users quantify and analyze expression patterns in mosaic tissues.

Expression patterns are typically identified by comparing the fluorescence of target gene reporters between groups of cells. Fly-QMA helps quantify these differences in reporter expression. Quantification entails measuring fluorescence intensities by analyzing images of imaginal discs. These measurements may then used to detect and analyze spatial patterns within each image.

Given confocal microscopy data, Fly-QMA facilitates:

  - **Segmentation.** Detect cell nuclei within each image

  - **Measurement.** Quantify reporter expression levels

  - **Bleedthrough Control.** Correct for fluorescence bleedthrough

  - **Annotation.** Label distinct subpopulations of cells

  - **ROI Definition.** Manually define regions of interest

  - **Comparison.** Compare expression levels between subpopulations


To get started, simply ``pip install flyqma``


.. toctree::
   :hidden:
   :maxdepth: 2

   INSTALLATION <installation>
   GETTING STARTED <start>
   FILE STRUCTURE <filestructure>
   DOCUMENTATION <documentation>
   EXAMPLE USAGE <usage>
   CONTACT <contact>
