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
- CARS pipelines
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
    - prepare: corrected grid creation with disparity min/max estimation
    - compute_dsm: dsm generation chaining time consuming steps (rectification, dense matching, triangulation, rasterization)
* steps:
    - rectification:
    - matching:
    - triangulation:
    - rasterization
