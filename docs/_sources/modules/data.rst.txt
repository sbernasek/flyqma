DATA
====

``clones.data`` provides three levels of organization for managing cell measurement data:

  1. ``Layer``: a 2D RGB cross section of an eye disc

  2. ``Stack``: a set of layers obtained from the same eye disc

  3. ``Experiment``: a collection of eye discs obtained under similar conditions


Layers
------

Layers are 2D RGB cross sections of an eye disc.

.. automodule:: clones.data.layers
   :members:


Stacks
------

Stacks are sets of layers obtained from the same eye disc.

.. automodule:: clones.data.stacks
   :members:


Experiments
-----------

Experiments are collections of stacks obtained under similar conditions.

.. automodule:: clones.data.experiments
   :members:

