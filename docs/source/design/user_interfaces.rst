===============
User interfaces
===============

CARS Main Python API
====================

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
===========================

The command line interface could be only corresponding to CARS Main Python API one to one

1. cars -i inputs -o output -c conf ? or cars conf.json (or yaml if hydra)

2. cars prepare

3. cars compute


CARS 3D Functional User interfaces
==================================

Steps
-----


1.  rectification
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
