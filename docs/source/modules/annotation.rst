.. image:: ../graphics/Northwestern_purple_RGB.png
   :width: 30%
   :align: right
   :alt: nulogo
   :target: https://amaral.northwestern.edu/


Annotation Module
=================

``flyqma.annotation`` provides several tools for labeling distinct subpopulations of cells within an image. Subpopulations are identified on the basis of their clonal marker expression level using a novel unsupervised classification strategy. Please see the Fly-QMA manuscript for a detailed description of the annotation strategy and its various parameters.

.. automodule:: flyqma.annotation


Mixture Models
--------------

Tools for fitting univariate and bivariate gaussian mixture models.

.. automodule:: flyqma.annotation.mixtures
   :members:


Model Selection
---------------

Tools for statistical model selection.

.. automodule:: flyqma.annotation
   :members:


Label Assignment
----------------

Tools for unsupervised classification of cell measurements.

.. automodule:: flyqma.annotation.classification
   :members:


Spatial Analysis
----------------

Tools for analyzing the 2D spatial arrangement of cells.

.. automodule:: flyqma.annotation.spatial
   :members:
