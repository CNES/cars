======================
CARS Main architecture
======================




Data representation
===================

CARS dataset image
------------------

core/dataset.py

geometric model
---------------
CARS representation ? link with modularity of geometric loaders. 
One CARS internal representation ? external ?




Module organization.
====================

- CARS main CLI API
- cars pipelines
- CARS steps :
- CARS core :
- CARS conf :

Here is the file organization accordingly:

* __init__.py : contains main CARS API ?
* cars.py  : main CARS cli loader
* cluster : all cluster scheduling strategy and optimization process
* conf: CARS configuration capabilities
* core: CARS General usage functions
* pipelines:
    - prepare: first general 3D preparation pipeline
    - compute_dsm: Main 3D pipeline
* steps:
    - rectification:
    - matching:
    - triangulation:
    - rasterization
