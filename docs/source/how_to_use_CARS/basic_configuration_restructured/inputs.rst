.. _inputs:

Inputs
======

CARS can be entered either with Sensor Images or with DSM.

Running CARS with sensor images as inputs
-----------------------------------------

The standard configuration uses sensor images as inputs. Additional parameters can be used in inputs configuration :

+----------------------------+---------------------------------------------------------------------+-----------------------------+----------------------+----------+
| Name                       | Description                                                         | Type                        | Default value        | Required |
+============================+=====================================================================+=============================+======================+==========+
| *sensors*                  | Stereo sensor images                                                | dict                        | No                   | Yes      |
+----------------------------+---------------------------------------------------------------------+-----------------------------+----------------------+----------+
| *pairing*                  | Association of sensor image to create pairs                         | list of pairs of *sensors*  | No                   | Yes (*)  |
+----------------------------+---------------------------------------------------------------------+-----------------------------+----------------------+----------+
| *initial_elevation*        | Low resolution DEM                                                  | string or dict              | No                   | No       |
+----------------------------+---------------------------------------------------------------------+-----------------------------+----------------------+----------+
| *roi*                      | Region Of Interest: Vector file path or GeoJson dictionary          | string or dict              | None                 | No       |
+----------------------------+---------------------------------------------------------------------+-----------------------------+----------------------+----------+

(*) `pairing` is required if there are more than two sensors (see pairing section below)

.. tabs::

    .. tab:: Sensors

        For each sensor image, give a particular name (what you want):

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/inputs_sensor_image

        +-------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------+---------------+----------+
        | Name              | Description                                                                                                                     | Type           | Default value | Required |
        +===================+=================================================================================================================================+================+===============+==========+
        | *image*           | Path to the image or dictionary readable by a sensor loader (see :ref:`advanced configuration`)                                 | string, dict   |               | Yes      |
        +-------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------+---------------+----------+
        | *geomodel*        | Path to the geomodel and plugin-specific attributes                                                                             | string, dict   |               | No       |
        +-------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------+---------------+----------+
        | *mask*            | Path to the binary mask                                                                                                         | string         | None          | No       |
        +-------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------+---------------+----------+
        | *classification*  | Path to the multiband binary classification image or dictionary readable by a sensor loader (see :ref:`advanced configuration`) | string         | None          | No       |
        +-------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------+---------------+----------+

        .. note::
            - *mask*: This image is a binary file. By using this file, the 1 values are not processed, only 0 values are considered as valid data.
            - *classification*: This image is a multiband binary file.
            - Please, see the section :ref:`convert_image_to_binary_image` to make binary *mask* image or binary *classification* image with 1 bit per band.
            - *geomodel*: If the geomodel file is not provided, CARS will use the RPC loaded with rasterio opening *image*.
            - It is possible to add sensors inputs while using depth_maps or dsm inputs

    .. tab:: Pairing

        The `pairing` attribute defines the pairs to use, using sensors keys used to define sensor images.

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/inputs_sensor_image_pairing

        This attribute is required when there are more than two input sensor images. If only two images ares provided, the pairing can be deduced by cars, considering the first image defined as the left image and second image as right image.

    .. tab:: Initial Elevation

        The attribute contains all informations about initial elevation: dem path, geoid path and default altitudes. 
        The initial elevation provided by a user will be used for the first resolution. 
        Subsequent resolutions will use the `DEMs Min/Median/Max` generated by the previous resolution as their initial elevation.

        +-----------------------+----------------------------------------------------------------------------+--------+----------------------+----------------------+----------+
        | Name                  | Description                                                                | Type   | Available value      | Default value        | Required |
        +=======================+============================================================================+========+======================+======================+==========+
        | *dem*                 | Path to DEM file (one tile or VRT with concatenated tiles)                 | string |                      | None                 | No       |
        +-----------------------+----------------------------------------------------------------------------+--------+----------------------+----------------------+----------+
        | *geoid*               | Path to geoid file                                                         | string |                      | CARS internal geoid  | No       |
        +-----------------------+----------------------------------------------------------------------------+--------+----------------------+----------------------+----------+
        | *altitude_delta_min*  | Constant delta in altitude (meters) between *dem_median* and *dem_min*     | int    | should be > 0        | None                 | No       |
        +-----------------------+----------------------------------------------------------------------------+--------+----------------------+----------------------+----------+
        | *altitude_delta_max*  | Constant delta in altitude (meters) between *dem_max* and *dem_median*     | int    | should be > 0        | None                 | No       |
        +-----------------------+----------------------------------------------------------------------------+--------+----------------------+----------------------+----------+

        See section :ref:`download_srtm_tiles` to download 90-m SRTM DEM.
        If no DEM path is provided, the `SIFT` matches will be used to reduce the disparity for the first resolution.

        If no geoid is provided, the default cars geoid is used (egm96).

        If no altitude delta is provided, the `dem_min` and `dem_max` generated with sparse matches will be used.

        The altitude deltas are used following this formula:

        .. code-block:: python

            dem_min = initial_elevation - altitude_delta_min
            dem_max = initial_elevation + altitude_delta_max

        .. warning::  DEM path is mandatory for the use of the altitude deltas.


        Initial elevation can be provided as a dictionary with a field for each parameter, for example:

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/inputs_initial_elevation_1

        Alternatively, it can be set as a string corresponding to the DEM path, in which case default values for the geoid and the default altitude are used.

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/inputs_initial_elevation_2

        Note that the `geoid` parameter in `initial_elevation` is not the geoid used for output products generated after the triangulation step
        (see output parameters).

        Elevation management is tightly linked to the geometry plugin used. See :ref:`plugins` section for details

    .. tab:: ROI

        A terrain ROI can be provided by the user. It can be either a vector file (Shapefile for instance) path,
        or a GeoJson dictionary. These structures must contain a single Polygon or MultiPolygon. Multi-features are
        not supported. Instead of cropping the input images, the whole images will be used to compute grid correction
        and terrain + epipolar a priori. Then the rest of the pipeline will use the given roi. This allow better correction 
        of epipolar rectification grids.


        Example of the "roi" parameter with a GeoJson dictionary containing a Polygon as feature :

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/inputs_roi_1

        If the *debug_with_roi* advanced parameter (see dedicated tab) is enabled, the tiling of the entire image is kept but only the tiles intersecting
        the ROI are computed.

        MultiPolygon feature is only useful if the parameter *debug_with_roi* is activated, otherwise the total footprint of the
        MultiPolygon will be used as ROI.

        By default epsg 4326 is used. If the user has defined a polygon in a different reference system, the "crs" field must be specified.

        Example of the *debug_with_roi* mode utilizing an "roi" parameter of type MultiPolygon as a feature and a specific EPSG.

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/inputs_roi_2

        Example of the "roi" parameter with a Shapefile

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/inputs_roi_3

Running CARS with DSM as inputs
-------------------------------

CARS can also be launched with DSM as inputs. The pipeline launched is just a merging of the DSM.

+----------------------------+--------------------------------------------------------------------------------+-----------------------------+----------------------+----------+
| Name                       | Description                                                                    | Type                        | Default value        | Required |
+============================+================================================================================+=============================+======================+==========+
| *dsm*                      | List of DSM to merge                                                           | dict                        | No                   | Yes      |
+----------------------------+--------------------------------------------------------------------------------+-----------------------------+----------------------+----------+
| *roi*                      | Region Of Interest: Vector file path or GeoJson dictionary                     | string or dict              | None                 | No       |
+----------------------------+--------------------------------------------------------------------------------+-----------------------------+----------------------+----------+
| *initial_elevation*        | Low resolution DEM (used for DSM filling)                                      | string or dict              | No                   | No       |
+----------------------------+--------------------------------------------------------------------------------+-----------------------------+----------------------+----------+
| *sensors*                  | Stereo sensor images used to generate the DSM                                  | dict                        | No                   | No       |
+----------------------------+--------------------------------------------------------------------------------+-----------------------------+----------------------+----------+
| *pairing*                  | Association of sensor images used to generate the DSM                          | list of pairs of *sensors*  | No                   | No (*)   |
+----------------------------+--------------------------------------------------------------------------------+-----------------------------+----------------------+----------+

(*) `pairing` is required if `sensors` parameter is set and contains more than two sensors

For each DSMS, give a particular name (what you want):

.. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/inputs_dsms

These input files can be generated by running CARS with `product_level: ["dsm"]` and `auxiliary` dictionary filled with desired auxiliary files

.. note::

    Only one method for performance map generation should have been selected: only two dimensions rasters for `dsm_inf*.tif`, `dsm_sup*.tif`, `performance_map.tif` are supported.
    
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| Name                       | Description                                                       | Type           | Default value | Required |
+============================+===================================================================+================+===============+==========+
| *dsm*                      | Path to the dsm file                                              | string         |               | Yes      |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *weights*                  | Path to the weights file                                          | string         |               | Yes      |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *image*                    | Path to the texture file                                          | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *classification*           | Path to the classification file                                   | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *mask*                     | Path to the mask file                                             | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *filling*                  | Path to the filling file                                          | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *performance_map*          | Path to the performance_map file                                  | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *contributing_pair*        | Path to the contributing_pair file                                | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *dsm_inf*                  | Path to the dsm_inf file                                          | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *dsm_sup*                  | Path to the dsm_sup file                                          | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *dsm_mean*                 | Path to the dsm_mean file                                         | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *dsm_std*                  | Path to the dsm_std file                                          | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *dsm_inf_mean*             | Path to the dsm_inf_mean file                                     | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *dsm_inf_std*              | Path to the dsm_inf_std file                                      | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *dsm_sup_mean*             | Path to the dsm_sup_mean file                                     | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *dsm_sup_std*              | Path to the dsm_sup_std file                                      | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *dsm_n_pts*                | Path to the dsm_n_pts file                                        | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *dsm_pts_in_cell*          | Path to the dsm_pts_in_cell file                                  | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
| *ambiguity*	             | Path to the ambiguity                                             | string         |               | No       |
+----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
