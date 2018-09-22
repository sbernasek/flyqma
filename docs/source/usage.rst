Example Usage
=============

Clones provides a wide range of functionality for measuring and analyzing gene expression in eye clones. A brief introduction to the major functionalities is provided here. For detailed use cases please see the :ref:`submodules <modules>` documentation.


Loading image data
------------------

Load an experiment, stack, or layer from a directory containing subdirectories of ``.tif`` files:

.. code-block:: python

    from clones.data import experiments

    # load a specific experiment
    path = './data/clones/yanE833'
    experiment = experiments.Experiment(path)

    # load a specific stack
    stack_id = 0
    stack = experiment[stack_id]

    # load a specific layer
    layer_id = 0
    layer = stack[layer_id]



Measuring expression
--------------------

Segment an image layer, measure the segment properties, and save the results:


.. code-block:: python

    layer.segment()

    layer.save()


Correcting fluorescence bleedthrough
------------------------------------

Perform bleedthrough correction:

.. code-block:: python

    from clones.bleedthrough.correction import LayerCorrection

    correction = LayerCorrection(layer, niters=50)
    correction.show_correction()
    correction.save()


Using the cell selection GUI
----------------------------

Launch the cell selection GUI for a fully loaded image stack:

.. code-block:: python

    from clones.selection.gui import GUI

    stack = experiment.load_stack(stack_ind, full=True)

    gui = GUI.load(stack)


Aggregating measurement data
----------------------------

Aggregate all measurement data for an experiment:

.. code-block:: python

    data = experiment.aggregate_measurements()


**Further Examples**

For detailed usage examples, please refer to the `code <https://github.com/sebastianbernasek/pnt_yan_ratio>`_ used to generate the figures in our manuscript.
