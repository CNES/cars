=================
CARS Requirements
=================

Functional requirements
------------------------
Here are some high level 3D requirements:

- Be able to generate a DSM from stereo images in geotif format.
- Be able to generate a 3D point cloud in classical format (?) from stereo images.
- Be able to generate a radiometric color image superposable with DSM
- Be able to use input multi functional masks to use in the 3D pipeline (typically water masks)

- Automatic pipeline: Be able to run a 3D pipeline with the minimal parameters
- Configurable pipeline: Be able to configure several internal steps with common 3D parameters (example: disparity range)
- Open pipeline: be able to have insights on internal algorithms and intermediate results and stats.
- Be able to launch easily each 3D step with a stabilized API.



Input data requirements
-----------------------
Here are some input sensor image requirements:

* Be able to open Pléiades (PHR), Spot 6/7, DigitalGlobe, ...??
* Be able to open tif or jp2 image format.

Here are some input sensor geometric model requirement

* Be able to open RPC models, grid models or physical model?
* Be able to open internal geometric?

Other input data:

* Be able to open a reference input DEM to optimize the pipeline.
* Be able to open masks (format?) and multi class definition configuration.

Output data requirements
------------------------

On DSM:

- geotif format

On Point cloud:

- which format?


Code requirements
------------------

- Python3.6+
- Follow PEP standards
- Follow continuous integration standards and version control.

- Ease new developers integration in code. Debug mode

- Documentation: user, developer, advanced, auto API.

- Complete testing of all cases: unit tests, reliability tests, end2end test.

- Standard Code quality check with common python tools (isort, black, flake8, pylint, mypy).

- Have an amount of helpful comments, complete docstrings for all functions and classes.

Design requirements
-------------------

- Easy install in software and python standards
- Easy configuration
- Have a clean and efficient code organization coherent with design architecture.
- Keep functional steps independent to ease evolution and simplify maintenance.
- Be able to activate only some parts with one common configuration file.
- Be able to have intermediate results when needed.



Modularity
----------

Libraries modularities
%%%%%%%%%%%%%%%%%%%%%%

Here are several needs for future CARS:

- Geometric core library: be able to have OTB, shareloc, libGEO, ... internal and by plugins
- Matching step: be able to call several matching tool. Pandora has to be called in a generic way with clean API.
- Input data library: be able to input several type of images. Only rasterio possible? or plugins also here?
- Cluster Scheduling libraries: be able to use several load distribution libraries (dask, multiprocessing, sequential, ...)

Other?

Naming: ``loader`` and ``loader_type``

Pipeline Steps modularity
%%%%%%%%%%%%%%%%%%%%%%%%%

Another modularity is the possibility to include other code between steps.

TODO
Maybe with  the possibility to change static call graph?  and sub functions between steps?
Maybe with some possibilities to add plugins in pipeline between steps?

Naming: ``plugin`` and ``plugin_type``


Shareable
---------
The software has to be easily shareable and developed by other people.
Needs clean design, documentation, examples, notebooks, ...


TODO: add missing requirements.
