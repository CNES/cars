.. _software_conception:

.. role:: raw-html(raw)
   :format: html

:raw-html:`<h1>Software</h1>`



CARS design aims a modular and customizable framework for multiview 3d reconstruction.
This design is organized around key concepts described in this section.

.. warning::

    Under construction with CARS design evolution.


The CARS framework can be introduced by the following diagram:

.. figure:: ../../images/design_overview.png
    :align: center
    :alt: Cars Framework

This section presents one by one the CARS key concepts and their interaction.

    * cars_dataset Input and output object of an application. Contains a calculated and potentially tiled data.
    * application: Algorithmic methods that takes
    * Orchestrator: It instantiates and interfaces with the cluster to which it provides the tasks to be processed. It is responsible for writing the data calculated by the cluster on the fly.
    * plugin: library or external tools providing specific 3d functions. Under heavy reconstruction !
    * **Pipeline**: A chain of applications ( 3d reconstruction steps) from input to output with intermediate data (CarsDataset) controlled by orchestrator;


.. tabs::

    .. tab:: CarsDataset

        .. include:: carsdataset.rst


    .. tab:: Application

        .. include:: application.rst


    .. tab:: Orchestrator

        .. include:: orchestrator.rst



    .. tab:: Plugin

        .. include:: plugin.rst




Detailed interaction between concepts
=====================================

Now that all the concepts have been presented in details, we can draw a more technical diagram:

.. figure:: ../../images/orchestrator_app_cluster_dataset.png
    :align: center
    :alt: Overview concepts details