.. _creating_a_plugin:

=================
Creating a plugin
=================

.. _plugins:

Implement CARS Concepts
=======================

This section describes optional plugin possibilities of CARS. 

.. note::
    
    Work in progress !

Plugins can be used to create new pipelines, overload applications in CARS pipelines or overload geometry functions



Pipeline plugin
---------------

A pipeline plugin will add a new pipeline in CARS, with existing applications or new applications brought by the plugin.
If the plugin installed is a pipeline plugin, it can be used by specifying the "pipeline" parameter in your CARS configuration file.
For example :

.. include-cars-config:: ../example_configs/developer_guide/creating_a_plugin/pipeline_plugin

New applications can be parametrized in "applications" section of the configuration file.

Application plugin
------------------

An application plugin will overload an existing application in CARS with a new method.
If the plugin installed is an application plugin, it can be used by adding the application in your CARS configuration file.
For example a point cloud denoising plugin can be used as follow :

.. include-cars-config:: ../example_configs/developer_guide/creating_a_plugin/application_plugin

Geometry plugin
---------------

If the plugin installed is a pipeline plugin, it can be used by specifying the "geometry_plugin" parameter in your CARS configuration file.
For example :

.. include-cars-config:: ../example_configs/developer_guide/creating_a_plugin/geometry_plugin

Sensor loader plugin
--------------------

Sensors loader plugins can be used by setting the "loader" parameter in "image" and "classification" attributes of the input configuration.
The loader defines several plugin-specific parameters

.. include-cars-config:: ../example_configs/developer_guide/creating_a_plugin/sensor_loader_plugin

A sensor loader plugin is a class that overrides the `SensorLoaderTemplate` class. It must define two methods : 
 - check_conf : check the plugin-specific parameters
 - set_pivot_format : transform the input configuration into a configuration readable by CARS using pivot format and set it in class attribute `self.pivot_format`. Specifications of pivot format are explained on :ref:`input`.


.. _develop_a_plugin:

Develop a plugin
----------------



.. tabs::

    .. tab:: Geometry Plugin

        Geometry plugins aim to enable the use of different geometry libraries, typically `Shareloc <https://github.com/CNES/shareloc>`_ to perform CARS geometric operations which require the interpretation of the geometric models of the pairs to process.

        Those operations are:

        * The epipolar grids computation
        * The direct localization operation
        * The lines of sight triangulation

        `SharelocGeometry` is an internal geometry plugin used in the baseline installations of CARS.
        In the current state, Shareloc has to be installed when using CARS as this class is imported in the step module.

        *Geometry abstract class*

        The CARS abstract geometry class, named `AbstractGeometry`, is defined in the core geometry module  (`cars/core/geometry/__init__.py`).
        Considering the geometry operations, CARS has its own internal implementation of the CARS abstract class using Shareloc. External plugins can be used if they are registered correctly :

        .. code-block:: python

            from cars.core.geometry.abstract_geometry import AbstractGeometry

            @AbstractGeometry.register_subclass("GeometryPluginName")
            class GeometryPluginName(AbstractGeometry):
                ...

        In order to make an external plugin work with CARS, it is mandatory to use the entry point `cars.plugins` at the setup of the package to register the AbstractGeometry object.
        For example, if the AbstractGeometry object is defined in file `cars_geometry_plugin_name.cars_geometry_plugin_name`, this code has to present in the file `cars_geometry_plugin_name.setup.py`

        .. code-block:: python

            setup(
                entry_points={
                    "cars.plugins": [
                        "plugin=cars_geometry_plugin_name.cars_geometry_plugin_name :GeometryPluginName"
                    ]
                },
            )

        *Mandatory methods*

        Currently, the `AbstractGeometry` class requires the implementation of the following mandatory methods and properties:

        * `conf_schema` which specify the user input json schema required by the geometric library.

        .. code-block:: python

            def conf_schema(self):
                """
                Returns the input configuration fields required by the geometry plugin
                as a json checker schema. The available fields are defined in the
                cars/conf/input_parameters.py file

                :return: the geo configuration schema
                """

        * `check_products_consistency` which check if the geometrical model filled by the user is readable by the geometric library.

        .. code-block:: python

            def check_products_consistency(cars_conf) -> bool:
                """
                Test if the product is readable by the geometry plugin

                :param: cars_conf: cars input configuration
                :return: True if the products are readable, False otherwise
                """

        * `triangulate` which is a method performing the triangulation from a disparity map or a set of matching points (mode parameter).

        .. code-block:: python

            def triangulate(
                sensor1,
                sensor2,
                geomodel1,
                geomodel2,
                mode: str,
                matches: Union[xr.Dataset, np.ndarray],
                grid1: str,
                grid2: str,
                roi_key: Union[None, str] = None,
            ) -> np.ndarray:
                """
                Performs triangulation from cars disparity or matches dataset

                :param sensor1: path to left sensor image
                :param sensor2: path to right sensor image
                :param geomodel1: path and attributes for left geomodel
                :param geomodel2: path and attributes for right geomodel
                :param mode: triangulation mode
                       (constants.DISP_MODE or constants.MATCHES)
                :param matches: cars disparity dataset or matches as numpy array
                :param grid1: path to epipolar grid of img1
                :param grid2: path to epipolar grid of image 2
                :param roi_key: dataset roi to use
                       (can be cst.ROI or cst.ROI_WITH_MARGINS)
                :return: the long/lat/height numpy array in output of the triangulation
                """

        * `generate_epipolar_grids` which generates the left and right epipolar grids from the images of the pair and their geometrical models.

        .. code-block:: python

            def generate_epipolar_grids(
                self,
                sensor1,
                sensor2,
                geomodel1,
                geomodel2,
                epipolar_step: int = 30,
            ) -> Tuple[
                np.ndarray, np.ndarray, List[float], List[float], List[int], float
            ]:
                """
                Computes the left and right epipolar grids

                :param sensor1: path to left sensor image
                :param sensor2: path to right sensor image
                :param geomodel1: path to left geomodel
                :param geomodel2: path to right geomodel
                :param epipolar_step: step to use to construct the epipolar grids
                :return: Tuple composed of :

                    - the left epipolar grid as a numpy array
                    - the right epipolar grid as a numpy array
                    - the left grid origin as a list of float
                    - the left grid spacing as a list of float
                    - the epipolar image size as a list of int \
                    (x-axis size is given with the index 0, y-axis size with index 1)
                    - the disparity to altitude ratio as a float
                """

        * `direct_loc` which performs direct localization operations.

        .. code-block:: python

            def direct_loc(
                self,
                sensor,
                geomodel,
                x_coord: list,
                y_coord: list,
                z_coord: list = None
            ) -> np.ndarray:
                """
                For a given image points list, compute the latitudes, longitudes, altitudes

                Advice: to be sure, use x,y,z list inputs only

                :param sensor: path to sensor image
                :param geomodel: path and attributes for geomodel
                :param x_coord: X Coordinates list in input image sensor
                :param y_coord: Y Coordinates list in input image sensor
                :param z_coord: Z Altitude coordinates list to take the image
                :return: Latitude, Longitude, Altitude coordinates list as a numpy array
                """

        Where `constants` corresponds to the `cars/core/constants.py` module.

        *Available methods*

        Some methods are available in the `AbstractGeometry` class that might be useful for any geometry plugin which would only perform the triangulation using sensor coordinates.
        CARS' API only provides as inputs of the geometry plugin triangulation method the epipolar coordinates for each image of the pair. Thus the `matches_to_sensor_coords` method enables any plugin to convert those coordinates into the corresponding sensor ones.

        `AbstractGeometry` implements the method `image_envelope`. It computes the ground footprint of an image in sensor geometry by projecting its four corners using the direct localization method. This method can be overloaded by any geometry plugin if necessary.


    .. tab:: Application Plugin

    .. tab:: Pipeline Plugin



Build the project Correctly
===========================

Installation
------------

To install a plugin, simply use pip inside your CARS environment :

.. code-block:: console

    source venv/bin/activate
    pip install plugin_name
