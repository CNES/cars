============
CARS Design
============

**DOCUMENT IN PROGRESS**

CARS is a new 3D tool with refactoring needs.
This document does not describe the current CARS design but the intended one.
The document is updated with refactoring design evolution.
Don't hesitate to complete with refactoring developments.

The main goals to this new design are:

- Stabilize user interfaces to CARS in long term
- Minimize the maintenance of the code with good development python practises
- Share the global view of future CARS to ease development
- Simplify developments


CARS Context
============

CARS is a dedicated and open source 3D tool to produce Digital Surface Models from satellite imaging by photogrammetry.
This Multiview stereo pipeline is intended for massive DSM production with a robust and performant design.

This tool has indeed two main targets:

- be a performant and stable tool in projects ground segment: stability, performance.
- be an engineering tool to test new algorithms for study : evolutivity, modularity, documentation.

For this two contexts, CARS has to be well designed and refactored.

.. toctree::
    :maxdepth: 2

    definitions
    requirements
    user_interfaces
    architecture
    cars_main
    cars_conf
    cars_core
    cars_pipelines
    cars_steps
    cars_cluster
    cars_loaders
    cars_plugins
    cars_roi
