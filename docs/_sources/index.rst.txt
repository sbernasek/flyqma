.. image:: graphics/Northwestern_purple_RGB.png
   :width: 30%
   :align: right
   :alt: nulogo
   :target: https://amaral.northwestern.edu/


Fly-QMA
=======


**Fly-QMA** is part of the **NU FlyEye** platform for quantitative analysis of *Drosophila* imaginal discs. The package enables Quantitative Mosaic Analysis (QMA) - that is, it helps users quantify and analyze expression patterns in mosaic tissues.

Expression patterns are typically identified by comparing the intensities of fluorescent reporters between groups of cells. Fly-QMA uses computer vision to quantify these differences in reporter expression by inferring them from microscope images. The measurements may then used to detect and analyze spatial patterns that might otherwise go unnoticed. Check out our `manuscript <https://doi.org/10.1101/775783>`_ for a detailed explanation of the algorithms underlying Fly-QMA.

Given microscopy data, Fly-QMA facilitates:

  - Image **segmentation** to detect cell nuclei

  - Quantitative **measurement** of reporter expression levels

  - Automated **bleedthrough control** for enhanced measurement accuracy

  - Automated **annotation** of clonal patch patterns

  - Manual **ROI definition** using a built-in GUI

  - **Statistical analysis** of expression levels and tissue morphology

To get started, simply ``pip install flyqma`` then skim through our :ref:`beginners guide <start>`.

.. toctree::
   :hidden:
   :maxdepth: 2

   INSTALLATION <installation>
   GETTING STARTED <start>
   FILE STRUCTURE <filestructure>
   INTEGRATIONS <integration>
   ROI DEFINITION <roi>
   DOCUMENTATION <documentation>
   PROJECTS <examples>
   CONTACT <contact>
