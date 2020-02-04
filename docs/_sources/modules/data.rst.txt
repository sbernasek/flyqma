.. image:: ../graphics/Northwestern_purple_RGB.png
   :width: 30%
   :align: right
   :alt: nulogo
   :target: https://amaral.northwestern.edu/


Data Module
===========

``flyqma.data`` provides three levels of organization for managing cell measurement data:

  1. ``Layer``: a 2D cross sectional image of an eye disc

  2. ``Stack``: a set of layers obtained from the same eye disc

  3. ``Experiment``: a collection of eye discs obtained under similar conditions


Images
------

Images are 2D arrays of pixel intensities recorded within one or more fluorescence channels.

.. automodule:: flyqma.data.images
   :members:


Layers
------

Layers are 2D cross sectional images of an eye disc.

.. automodule:: flyqma.data.layers
   :members:


Stacks
------

Stacks are sets of layers obtained from the same eye disc.

.. automodule:: flyqma.data.stacks
   :members:


Experiments
-----------

Experiments are collections of stacks obtained under similar conditions.

.. automodule:: flyqma.data.experiments
   :members:


Silhouette Interface
--------------------

Fly-QMA provides several tools for seemlessly exchanging data with NU FlyEye Silhouette.

.. automodule:: flyqma.data.silhouette_read
   :members:

.. automodule:: flyqma.data.silhouette_write
   :members:

