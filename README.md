CARS
====

**CARS** is a dedicated and open source 3D tool to produce **Digital Surface Models** from satellite imaging by photogrammetry.
This Multiview stereo pipeline is intended for massive DSM production with a robust and performant design. 
CARS means CNES Algorithms to Reconstruct Surface (or Chaîne Automatique de Restitution Stéréoscopique in french)

It is composed of:

* A **Python API**, based on **xarray**, enabling to realize all the computation steps leading to a DSM.
* An **end-to-end processing chain** based on this API. It can be performed using **dask** (locally or on a cluster which has a GPFS centralized files storage) or **multiprocessing** libraries to distribute the computations.

Documentation 
=============

1. [Generalities](./docs/generalities.rst)
2. [Installation](./docs/install.rst)
3. [Command line usage](./docs/cli_usage.rst)
4. [Jupyter notebooks](./docs/notebooks.rst)

Contribution 
============
To do a bug report or a contribution, see the [**contribution guide**](CONTRIBUTING.md).

Changelog
=========
To know project evolution, see the [**Changelog**](CHANGELOG.md)

Licence
=======
See [the license](./LICENSE) for all legal issues concerning the use of CARS.

References
==========

- Youssefi D., Michel, J., Sarrazin, E., Buffe, F., Cournet, M., Delvit, J., L’Helguen, C., Melet, O., Emilien, A., Bosman, J., 2020. Cars: A photogrammetry pipeline using dask graphs to construct a global 3d model. IGARSS - IEEE International Geoscience and Remote Sensing Symposium.

- Michel, J., Sarrazin, E., Youssefi, D., Cournet, M., Buffe, F., Delvit, J., Emilien, A., Bosman, J., Melet, O., L’Helguen, C., 2020. A new satellite imagery stereo pipeline designed for scalability, robustness and performance. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences.