.. image:: ../graphics/Northwestern_purple_RGB.png
   :width: 30%
   :align: right
   :alt: nulogo
   :target: https://amaral.northwestern.edu/


Annotation Module
=================

``flyqma.annotation`` provides several tools for labeling distinct subpopulations of cells within an image. Subpopulations are identified on the basis of their clonal marker expression level using a novel unsupervised classification strategy. Please see the Fly-QMA manuscript for a detailed description of the annotation strategy and its various parameters.

.. automodule:: flyqma.annotation.labelers
   :members:

.. automodule:: flyqma.annotation.annotation
   :members:


Mixture Models
--------------

Tools for fitting univariate and bivariate gaussian mixture models.

.. automodule:: flyqma.annotation.mixtures.univariate
   :members:

.. automodule:: flyqma.annotation.mixtures.bivariate
   :members:

.. automodule:: flyqma.annotation.mixtures.visualization
   :members:


Model Selection
---------------

Tools for statistical model selection.

.. automodule:: flyqma.annotation.model_selection.univariate
   :members:

.. automodule:: flyqma.annotation.model_selection.bivariate
   :members:

.. automodule:: flyqma.annotation.model_selection.visualization
   :members:


Label Assignment
----------------

Tools for unsupervised classification of cell measurements.

.. automodule:: flyqma.annotation.classification.classifiers
   :members:

.. automodule:: flyqma.annotation.classification.kmeans
   :members:

.. automodule:: flyqma.annotation.classification.mixtures
   :members:

.. automodule:: flyqma.annotation.classification.visualization
   :members:


Spatial Analysis
----------------

Tools for analyzing the 2D spatial arrangement of cells.

.. automodule:: flyqma.annotation.spatial.triangulation
   :members:

.. automodule:: flyqma.annotation.spatial.graphs
   :members:

.. automodule:: flyqma.annotation.spatial.correlation
   :members:

.. automodule:: flyqma.annotation.spatial.infomap
   :members:

.. automodule:: flyqma.annotation.spatial.sampling
   :members:

.. automodule:: flyqma.annotation.spatial.timeseries
   :members:
