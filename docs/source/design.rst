============
CARS Design
============

**DOCUMENT IN PROGRESS**

CARS is a new tool which works but still needs refactoring.
This document does not describe the current CARS design but the intended one.

The main goals to this new design are:

- Stabilize user interfaces to CARS in long term
- Minimize the maintenance of the code with good development python practises
- Share the global view of future CARS to ease development
- Simplify developments


CARS Context
============

CARS is a dedicated and open source 3D tool to produce Digital Surface Models from satellite imaging by photogrammetry.
This Multiview stereo pipeline is intended for massive DSM production with a robust and performant design.

This tool has two main targets:
- be a performant and stable tool in projects ground segment: stability, performance, ...
- be an engineering tool to test new algorithms for study : evolutivity, modularity, documentation, ...

For this two contexts, CARS has to be well designed and refactored.

CARS Needs
==========

Modularity
----------

Here are several needs for future CARS:
- Geometric core library : be able to have OTB, shareloc, libgeo, ... internal and by plugins
- Matching step : be able to call several matching tool. Pandora has to be called in a generic way with clean API.
- Input data library : be able to input several type of images. Only rasterio possible ? or plugins also here ?

etc ...

This modularity has to be well designed and document so as another developer can easily add other possibilities.

Another modularity is the possibility to include other code
Maybe with  the possibility to change static call graph ?  and sub functions between steps ?
Maybe with some possibilities to add plugins in pipeline between steps ?


Shareable
---------
The software has to be easily shareable and developed by other people.
Needs clean design, documentation, examples, notebooks, ...


Others ?

CARS Definitions
================

- pipeline :
- step: a functional block (ie epipolar rectification) on one tile only
- tile: a subset of the entire input image to be processed on one worker only

- left_img : first image in image pair (rename img1?)
- right_img : second image in image pair (rename img2?)

User interfaces
===============

CARS Main Python API
--------------------

1. cars.run() : Main Python API for cars. Parameters

  - inputs : 1 pair only ?, N pairs, N images ?, mask, srtm, ? One format in a class ?
  - output : point_cloud ?, dsm ? an output.json file describing several elements ?
  - configuration: parameters of each step, core libs, ... kwargs ?


2. Sub pipelines

  2a. Prepare pipeline cars.pipelines.prepare.run()

  2b. Compute pipeline cars.pipelines.compute.run()

Prepare and compute have the same API


Code Example :





Questions:
- Inputs format : a json file / Input format Class ?
- Output format : not only an output directory but also a json file / Output format class ? the same than input incremented ?
- Configuration : separation static_conf and dynamic parameters ? have only one dynamic and static conf ?
- Rename compute_dsm in compute (not only dsm)

CARS Command Line Interface
---------------------------

The command line interface could be only corresponding to CARS Main Python API one to one

1. cars -i inputs -o output -c conf ? or cars conf.json (or yaml if hydra)

2. cars prepare

3. cars compute


CARS Functional User interfaces
===============================

Steps
-----


1.  epi_rectif
2.  matching
  2.a sparse_matching
  3.b dense_matching
3. triangulate
4. Filter point_cloud ?
5. Rasterize



Questions:
- Prepare steps AND compute steps ?
-

Core libraries
--------------

Scheduling
----------


CARS Main architecture
======================

- cars main CLI API
- cars pipelines
- CARS steps :
- CARS core :
- CARS conf :

Here is the file organization accordingly:




CARS Detailed architecture
==========================

CARS main algorithm
-------------------

Algo:
- cars.conf.init_conf()
- cars.run(input, output, conf)
  - client, cluster = cars.cluster.init_cluster()
  - graph = cars.cluster.init_graph(mode) %
  - asynchron writing launch ? pipeline write only on nodes by default ?
  - cars.pipelines.prepare(input, output, *params?, cluster, graph)
  - asynchron writing launch ? pipeline write only on nodes by default ?
  - cars.pipelines.compute(input, output, *params?, cluster, graph)
     - output can be point_cloud or dsm depending on dag


Questions:
- write_point_cloud or write_dsm ? sub step in parallel ?



CARS steps
----------

- Epi_rectif

- Matching

- Triangulation

- Filter cloud

- Rasterize


Questoins :

- prepare steps ?

CARS core
---------
