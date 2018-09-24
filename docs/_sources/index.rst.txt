====================
Clones Documentation
====================

.. image:: graphics/header.png
   :width: 100%
   :align: center
   :alt: header

|
|
| Welcome to Clones!


Clones is a platform for analyzing protein expression patterns in the developing *Drosophila* eye. The package is particularly focused on comparing expression between distinct clonal subpopulations within the eye epithelium.

Expression patterns are typically identified by comparing the fluorescence of target gene reporters between groups of cells. Clones helps quantify these differences in reporter expression. Quantification entails measuring fluorescence intensities by analyzing experimentally collected images of fixed eye discs. Such measurements may then used to identify and compare distinct subpopulations within the eye field.

Given confocal microscopy data, clones facilitates:

  - **Segmentation:** automated detection of cell nuclei

  - **Measurement:**  quantification of reporter expression levels

  - **Bleedthrough Control:** correction of fluorescence bleedthrough

  - **Annotation:** automated labeling of clonal subpopulations

  - **Cell Selection:** manual selection of cells for comparison

  - **Comparison:** statistical comparison of subpopulations


The initial release is primarily limited to basic tools required to replicate `our study <https://github.com/sebastianbernasek/pnt_yan_ratio>`_ of Pnt and Yan expression during retinal patterning in *Drosophila*. We intend to expand this functionality in later releases.


.. toctree::
   :hidden:
   :maxdepth: 2

   INSTALLATION <installation>
   API DOCUMENTATION <modules>
   GETTING STARTED <gettingstarted>
   EXAMPLE USAGE <usage>


Indices and tables
------------------

* :ref:`search`

