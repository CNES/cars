============
CARS loaders
============

CARS modularities aim to:

- Be able to use several geometry libraries
- Be able to use several matching libraries
- Be able to use several opening image tools
- Be able to use several cluster scheduling strategy/tools

The goals are to:

- answer several projects and user goals
- ease 3D studies and definition/test of new algorithms
- simplify evolution when a library becomes obsolete

In CARS, we call "loader" the generic way to handle those modularities.

The loaders implementations can be internal or external.

General loader design
=====================

The loaders are python classes that follows a specific CARS implementation. This is done by using abstract classes and a mechanism of registration as defined in the next sub-section.
The registration can be both internal or external. In the latter case the loader has to be installed in the user environment.

The several loaders are declined from this general way.

Abstract classes
----------------

In order to enable the use of loaders, an abstract class mechanism is used in CARS. Those classes are placed in the CARS' `core` module due to the cross-sectional aspect of the loaders.

An abstract class is a high-level class which specifies and unifies the required methods and properties to implement (i.e. the methods and properties which will be called by CARS along with their required interfaces).
Those methods and properties can be identified by the decorator `@abstract_method`.

Each loader have to inherit from the main CARS' abstract class and registered itself to it. For example, a geometry loader would be implemented as a class as follow:

.. code-block:: python

    from cars.core.geometry import AbstractGeometry

    @AbstractGeometry.register_subclass("GeometryLoaderName")
    class GeometryLoaderName(AbstractGeometry):
        ...


Loaders usage in CARS
---------------------

The abstract class instantiation is done in the static configuration module. The loader is defined as a global parameter which enable a single instantiation of the object in a given environment. thus, if the loader has to be used on worker, an abstract class object will be instantiated on each worker.

Each time a loader operation has to be called in a CARS step, the abstract class object of the static configuration is retrieved and the required method or property is called in the CARS' API function. This mechanism is transparent for CARS users as they only use the CARS' API..

The loaders to use in CARS are defined in a specific section of the static configuration:

.. code-block:: json

    "loaders":{
        "geometry": "GeometryLoaderName"
    }

Where `GeometryLoaderName` is the name of the loader class inheriting the main CARS abstract class as a string.

User configuration
------------------

Each loader may require specific user inputs and thus might modify the CARS user configuration file. In CARS the user configuration is checked against a json schema at the beginning of the `prepare` step in order to verify the inputs consistency as soon as possible. This mechanism has been adapted to take into account the loaders inputs consistency check as well.
In order to do so, each loader has to specify its own json schema via the abstract property (and thus mandatory) `conf_schema`. This schema will be merged with the basic CARS configuration schema. This minimal CARS configuration schema is defined by the variable INPUT_CONFIGURATION_SCHEMA in the `cars/conf/input_parameters.py` module.

Geometry loader
===============

Geometry loaders aim to enable the use of different geometry libraries, typically OTB, libGEO or Shareloc to perform CARS geometric operations which require the interpretation of the geometric models of the pairs to process.
Those operation are:

* The epipolar grids computation
* The direct localization operation
* The lines of sight triangulation

The `OTBGeometry` is the internal geometry loader used in the baseline installation of CARS. Thus, in the current state, the OTB has to be installed when using CARS as this class is imported in the step module. The OTBGeometry class might be used as a plugin in the future in order to remove the hard-coded import which imposes the installation of the OTB to be able to use CARS.

Geometry abstract class
-----------------------

The CARS' abstract geometry class, named `AbstractGeometry`, is defined in the core geometry module  (`cars/core/geometry/__init__.py`).
Considering the geometry operations, CARS have its own internal implementation of the CARS abstract class using the OTB. External loaders can be used if they are registered correctly :

.. code-block:: python

    from cars.core.geometry import AbstractGeometry

    @AbstractGeometry.register_subclass("GeometryLoaderName")
    class GeometryLoaderName(AbstractGeometry):
        ...

Mandatory methods
^^^^^^^^^^^^^^^^^
Currently, the `AbstractGeometry` class requires the implementation of the following mandatory methods and properties:

* `conf_schema` which specify the user inputs json schema required by the geometric library

.. code-block:: python

    def conf_schema(self):
        """
        Returns the input configuration fields required by the geometry loader
        as a json checker schema. The available fields are defined in the
        cars/conf/input_parameters.py file

        :return: the geo configuration schema
        """

* `check_products_consistency` which check if the geometrical model filled by the user is readable by the geometric library.

.. code-block:: python

    def check_products_consistency(cars_conf) -> bool:
        """
        Test if the product is readable by the geometry loader

        :param: cars_conf: cars input configuration
        :return: True if the products are readable, False otherwise
        """

* `triangulate` which is a method performing the triangulation from a disparity map or a set of matching points (mode parameter).

.. code-block:: python

    def triangulate(
        mode: str,
        matches: Union[xr.Dataset, np.ndarray],
        grid1: str,
        grid2: str,
        cars_conf,
        roi_key: Union[None, str] = None,
    ) -> np.ndarray:
        """
        Performs triangulation from cars disparity or matches dataset

        :param mode: triangulation mode
        (constants.DISP_MODE or constants.MATCHES)
        :param matches: cars disparity dataset or matches as numpy array
        :param grid1: path to epipolar grid of img1
        :param grid2: path to epipolar grid of image 2
        :param cars_conf: cars input configuration
        :param roi_key: dataset roi to use
        (can be cst.ROI or cst.ROI_WITH_MARGINS)
        :return: the long/lat/height numpy array in output of the triangulation
        """

* `generate_epipolar_grids` which generates the left and right epipolar grids from the images of the pair and their geometrical models

.. code-block:: python

    def generate_epipolar_grids(
        cars_conf,
        dem: Union[None, str] = None,
        default_alt: Union[None, float] = None,
        epipolar_step: int = 30,
    ) -> Tuple[
        np.ndarray, np.ndarray, List[float], List[float], List[int], float
    ]:
        """
        Computes the left and right epipolar grids

        :param cars_conf: cars input configuration
        :param dem: path to the dem folder
        :param default_alt: default altitude to use in the missing dem regions
        :param epipolar_step: step to use to construct the epipolar grids
        :return: Tuple composed of :
            - the left epipolar grid as a numpy array
            - the right epipolar grid as a numpy array
            - the left grid origin as a list of float
            - the left grid spacing as a list of float
            - the epipolar image size as a list of int
            (x-axis size is given with the index 0, y-axis size with index 1)
            - the disparity to altitude ratio as a float
        """

Where `constants` corresponds to the `cars/core/constants.py` module.

Available methods
^^^^^^^^^^^^^^^^^

Some methods are available in the `AbstractGeometry` class that might be useful for any geometry loader which would only perform the triangulation using sensor coordinates.
CARS' API only provides as inputs of the geometry loader triangulation method the epipolar coordinates for each image of the pair. Thus the `matches_to_sensor_coords` method enables any laoder to convert those coordinates into the corresponding sensor ones.

Matching loader
===============

TODO

Cluster loader
==============

TODO
