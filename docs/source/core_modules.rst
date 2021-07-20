.. _core_modules:

CARS uses external softwares to perform several operations which can be cross-sectional (i.e. required in different CARS steps). The latter can be divided into several categories:

* The geometry operations which require the interpretation of the geometric models of the pairs to process. Those operation are:

    * The epipolar grids computation
    * The image envelop computation
    * the direct localization operation
    * The lines of sight triangulation
    Those operation are currently performed using the OTB software

* The dense matching which is currently done using PANDORA

In order to enable the use of specific software, an abstract class mechanism has been implemented in CARS and placed in the CARS' `core` module due to the cross-sectional aspect of those operations.
An abstract class is a high-level class which specifies and unifies the required functions to implement (i.e. the functions which will be called by CARS along with their required interfaces). Those functions can be identified by the decorator `@abstract_method`.

Considering the geometry operations, CARS have its own internal implementation of the CARS abstract class using the OTB. A system of plugins will be implemented to enable the use of another software.
The dense matching abstract class is not implemented yet thus the use of PANDORA is mandatory in CARS.

Geometry abstract class
=======================

The CARS' abstract geometry class, named `AbstractGeometry`, is defined in `the core geometry module  <cars/core/geometry/__init__.py>`_.

Currently, the `AbstractGeometry` class requires the implementation of the following mandatory functions (see `the core geometry module <cars/core/geometry/__init__.py>`_ to know the interfaces):

* `triangulate` which is a function performing the triangulation from a disparity map or a set of matching points (`mode` parameter).

The other geometry operations listed in introduction will soon be implemented as **abstract methods** of the `AbstractGeometry` class.
Each time a geometry operation has to be performed in the CARS step, an instantiation of a `AbstractGeometry` object is done and the required method is called. This mechanism is transparent for CARS users.

Internal OTBGeometry class
--------------------------

The `OTBGeometry` is an **internal implementation** of the `AbstractGeometry` class and cannot be considered as a geometry plugin. It is used in the baseline installation of CARS. Thus, in the current state, the OTB has to be installed when using CARS as this class is imported in the step module. The `OTBGeometry` class might be used as a plugin in the future in order to remove the hard-coded import which imposes the installation of the OTB to be able to use CARS.

Use a geometry plugin
---------------------

This functionality is not available yet.

