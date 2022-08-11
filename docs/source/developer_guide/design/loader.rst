.. _loader:

Loader
======

Geometry loader
^^^^^^^^^^^^^^^^^

Geometry loaders aim to enable the use of different geometry libraries, typically OTB, libGEO or Shareloc to perform CARS geometric operations which require the interpretation of the geometric models of the pairs to process.
Those operation are:

* The epipolar grids computation
* The direct localization operation
* The lines of sight triangulation

The `OTBGeometry` is the internal geometry loader used in the baseline installation of CARS. Thus, in the current state, the OTB has to be installed when using CARS as this class is imported in the step module. The OTBGeometry class might be used as a plugin in the future in order to remove the hard-coded import which imposes the installation of the OTB to be able to use CARS.

Geometry abstract class
+++++++++++++++++++++++

The CARS' abstract geometry class, named `AbstractGeometry`, is defined in the core geometry module  (`cars/core/geometry/__init__.py`).
Considering the geometry operations, CARS have its own internal implementation of the CARS abstract class using the OTB. External loaders can be used if they are registered correctly :

.. code-block:: python

    from cars.core.geometry import AbstractGeometry

    @AbstractGeometry.register_subclass("GeometryLoaderName")
    class GeometryLoaderName(AbstractGeometry):
        ...

Mandatory methods

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

* `direct_loc` which performs direct localization operations

.. code-block:: python

    def direct_loc(
        conf,
        product_key: str,
        x_coord: float,
        y_coord: float,
        z_coord: float = None,
        dem: str = None,
        geoid: str = None,
        default_elevation: float = None,
    ) -> np.ndarray:
        """
        For a given image point, compute the latitude, longitude, altitude

        Advice: to be sure, use x,y,z inputs only

        :param conf: cars input configuration dictionary
        :param product_key: input_parameters.PRODUCT1_KEY or
        input_parameters.PRODUCT2_KEY to identify which geometric model shall
        be taken to perform the method
        :param x_coord: X Coordinate in input image sensor
        :param y_coord: Y Coordinate in input image sensor
        :param z_coord: Z Altitude coordinate to take the image
        :param dem: if z not defined, take this DEM directory input
        :param geoid: if z and dem not defined, take GEOID directory input
        :param default_elevation: if z, dem, geoid not defined, take default
        elevation
        :return: Latitude, Longitude, Altitude coordinates as a numpy array
        """

Where `constants` corresponds to the `cars/core/constants.py` module.

Available methods
+++++++++++++++++

Some methods are available in the `AbstractGeometry` class that might be useful for any geometry loader which would only perform the triangulation using sensor coordinates.
CARS' API only provides as inputs of the geometry loader triangulation method the epipolar coordinates for each image of the pair. Thus the `matches_to_sensor_coords` method enables any loader to convert those coordinates into the corresponding sensor ones.

`AbstractGeometry` implements the method `image_envelope`. It computes the ground footprint of an image in sensor geometry by projecting its four corners using the direct localization method. This method can be overloaded by any geometry loader if necessary.