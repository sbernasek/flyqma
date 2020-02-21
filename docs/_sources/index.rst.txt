.. image:: graphics/Northwestern_purple_RGB.png
   :width: 30%
   :align: right
   :alt: nulogo
   :target: https://amaral.northwestern.edu/


Fly-QMA
=======


**Fly-QMA** is part of the **NU FlyEye** platform for quantitative analysis of *Drosophila* imaginal discs. The package enables Quantitative Mosaic Analysis (QMA) - that is, it helps users quantify and analyze expression patterns in mosaic tissues.

Expression patterns are typically identified by comparing the intensities of fluorescent reporters between groups of cells. Fly-QMA uses computer vision to quantify these differences in reporter expression by inferring them from microscope images. The measurements may then used to detect and analyze spatial patterns that might otherwise go unnoticed. Check out our `manuscript <https://doi.org/10.1101/775783>`_ for a detailed explanation of the algorithms underlying Fly-QMA.

Given confocal microscopy data, Fly-QMA facilitates:

  - **Segmentation.** Detect cell nuclei within an image

  - **Measurement.** Quantify reporter expression levels

  - **Bleedthrough Control.** Correct for fluorescence bleedthrough

  - **Annotation.** Automatically label differing groups of cells

  - **ROI Definition.** Focus on particular regions of interest

  - **Comparison.** Compare expression levels between cells


To get started, simply ``pip install flyqma``


.. toctree::
   :hidden:
   :maxdepth: 2

   INSTALLATION <installation>
   GETTING STARTED <start>
   BASIC USAGE <usage>
   INTEGRATIONS <integration>
   FILE STRUCTURE <filestructure>
   DOCUMENTATION <documentation>
   CONTACT <contact>
