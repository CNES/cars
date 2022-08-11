.. _overview:

========
Overview
========

A 3D framework
**************

CARS is a dedicated and open source 3D framework to produce Digital Surface Models from satellite imaging by photogrammetry.

This framework is intended for massive :term:`DSM` production with a robust, performant and modular design.

.. figure:: ../images/cars_framework_diagram.png
    :width: 1000px
    :align: center
    :alt: CARS Framework
    
It is mainly composed of:

* :ref:`cli` configured through a :ref:`configuration` file.
* Pipelines:

  * :ref:`sensor_to_full_resolution_dsm_pipeline`: end-to-end processing pipeline from sensor images to full :term:`DSM`
  * *Sensor to low resolution_dsm*: subpart of the previous, pipeline based on sparse matches to generate a low resolution :term:`DSM`

* Python :ref:`3d_api`: composed of well known steps in Multiview Stereo Pipeline divided in :ref:`applications` used by pipelines.


Computing distribution
**********************

The chain have computing distribution capabilities and can use dask (local or distributed cluster) or multiprocessing libraries to distribute the computations.
The distributed cluster require centralized files storage and uses PBS scheduler only for now.

See :ref:`orchestrator_config` section
