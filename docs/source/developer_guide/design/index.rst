.. _cars_design:

===============
CARS Design
===============

CARS design aims a modular and customizable framework for multiview 3d reconstruction.
This design is organized around key concepts described in this section.

.. warning::
  
    Under construction with CARS design evolution.
    

The CARS framework can be introduced by the following diagram:

.. figure:: ../../images/design_overview.png
    :align: center
    :alt: Cars Framework

This section presents one by one the CARS key concepts and their interaction.

    * :ref:`cars_dataset` Input and output object of an application. Contains a calculated and potentially tiled data.
    * :ref:`application`: Algorithmic methods that takes
    * :ref:`Orchestrator`: It instantiates and interfaces with the cluster to which it provides the tasks to be processed. It is responsible for writing the data calculated by the cluster on the fly.
    * :ref:`plugin`: library or external tools providing specific 3d functions. Under heavy reconstruction !
    * **Pipeline**: A chain of applications ( 3d reconstruction steps) from input to output with intermediate data (CarsDataset) controlled by orchestrator; 

.. toctree::
    :maxdepth: 2

    carsdataset
    orchestrator
    cluster_mp
    application
    interaction
    plugin








