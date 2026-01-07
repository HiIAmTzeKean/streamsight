Welcome to Streamsight's documentation!
=======================================

.. image:: /_static/logo.png
    :align: center

Streamsight is an open-source python toolkit developed that provides a framework
which observes the context of time to accurately model offline setting to actual
real-world scenarios. We aim to provide API for the programmer to build and
evaluate recommendation systems.

The overall architecture of the package is shown in the figure below. We split
the toolkit into three main components: data handling, recommendation system,
and evaluation. The data handling component is responsible for loading and
preprocessing the data, the RecSys on implementing the recommendation algorithms
and the Evaluation for evaluating the recommendation algorithms.

.. image:: /_static/architecture.png
    :align: center

The demo notebooks can be found in the `examples` directory
`here <https://github.com/HiIAmTzeKean/Streamsight/tree/master/examples>`_.
The notebooks demonstrate how to use the toolkit to build a recommendation
system and evaluate them.

Contents
========

.. toctree::
    :maxdepth: 1
    :caption: Data handling

    streamsight.matrix
    streamsight.preprocessing
    streamsight.datasets
    streamsight.settings

.. toctree::
    :maxdepth: 1
    :caption: RecSys

    streamsight.algorithms

.. toctree::
    :maxdepth: 1
    :caption: Evaluation

    streamsight.evaluators
    streamsight.metrics

.. toctree::
    :maxdepth: 1
    :caption: Other supporting modules

    streamsight.registries
    streamsight.utils


Indices and References
==================

* :ref:`genindex`
* :ref:`ref-sheet`