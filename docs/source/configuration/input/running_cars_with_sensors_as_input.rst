Running CARS with sensor images as input
----------------------------------------

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

        **Basic configuration**

        For each sensor image, give a particular name (what you want):

        .. include-cars-config:: ../../example_configs/configuration/inputs_sensor_image_basic

        **Intermediate configuration**

        For each sensor, auxiliary files can be used : mask, classification, and geomodel if needed

        +-------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------+---------------+----------+
        | Name              | Description                                                                                                                     | Type           | Default value | Required |
        +===================+=================================================================================================================================+================+===============+==========+
        | *image*           | Path to the image or dictionary readable by a sensor loader                                                                     | string, dict   |               | Yes      |
        +-------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------+---------------+----------+
        | *geomodel*        | Path to the geomodel and plugin-specific attributes                                                                             | string, dict   |               | No       |
        +-------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------+---------------+----------+
        | *mask*            | Path to the binary mask                                                                                                         | string, dict   | None          | No       |
        +-------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------+---------------+----------+
        | *classification*  | Path to the classification image or dictionary readable by a sensor loader                                                      | string, dict   | None          | No       |
        +-------------------+---------------------------------------------------------------------------------------------------------------------------------+----------------+---------------+----------+

        In most cases, only the file path is required for each of these parameters and CARS will know how to read each file : 

        .. include-cars-config:: ../../example_configs/configuration/inputs_sensor_image_intermediate

        However for each parameter it is possible to set a dictionary with additional parameters on how to read the file. 
        
        **Advanced configuration**
        
        These parameters are described on the tabs below :

        .. tabs::

            .. tab:: Image

                The standard method for passing sensor images as inputs is to put only the path of the image. It works well with panchromatic images.

                If the images are multi-band, CARS will automatically perform the matching steps on the first band (for example if the image is RGB, CARS will correlate on the red band).

                However, CARS offers an advanced way to use images as inputs called **sensor loaders**. Sensor loaders can be useful in these cases :

                 - A multi-band image is passed as input and you want to control which band is used for correlation 
                 - A multi-band image is passed as input and you want to control which bands are used in the output orthorectified image.
                 - You want to concatenate several bands of a single image that are on separate files (for example a panchromatic image file and a RGB image file).

                At the moment only two sensor loaders are available in CARS : “basic” and “pivot”. To use them you juste have to pass a dictionary for the "image" parameter, with the key "loader".

                **Basic loader**

                The basic loader is the simplest way to define an image. The basic loader is the one used by default when only a path is given. However, it is possible to use the basic loader with a dictionary : 

                +----------------+-----------------------+--------+---------------+------------------+----------+
                | Name           | Description           | Type   | Default value | Available values | Required |
                +================+=======================+========+===============+==================+==========+
                | *loader*       | Name of sensor loader | str    | "basic_image" | "basic_image"    | No       |
                +----------------+-----------------------+--------+---------------+------------------+----------+
                | *path*         | File path             | str    |               |                  | Yes      |
                +----------------+-----------------------+--------+---------------+------------------+----------+
                | *no_data*      | No data value of file | int    | 0             |                  | No       |
                +----------------+-----------------------+--------+---------------+------------------+----------+

                An example is given below : 

                .. include-cars-config:: ../../example_configs/configuration/image_basic_loader_config
    
                **Pivot loader**

                The pivot loader allows the maximal level of configuration. To use the pivot loader, it is required to set the "loader" parameter in sensor loader configuration.

                +-----------------+---------------------------------------------------------------------------------------+--------+-------------------+------------------+----------+
                | Name            | Description                                                                           | Type   | Default value     | Available values | Required |
                +=================+=======================================================================================+========+===================+==================+==========+
                | *loader*        | Name of sensor loader                                                                 | str    | "basic_image"     | "pivot_image"    | Yes      |
                +-----------------+---------------------------------------------------------------------------------------+--------+-------------------+------------------+----------+
                | *bands*         | Dictionary listing for every band of the image, the corresponding file and band index | dict   |                   |                  | Yes      |
                +-----------------+---------------------------------------------------------------------------------------+--------+-------------------+------------------+----------+
                | *no_data*       | No data value of file                                                                 | int    | 0                 |                  | No       |
                +-----------------+---------------------------------------------------------------------------------------+--------+-------------------+------------------+----------+

                The `bands` dictionary have keys which correspond to name of bands. The name of bands is imposed by CARS : if the image has n bands, the name of the bands must be ["b0", "b1", ..., "b{n-1}"].
                Each key points to a dictionary with keys "path" and "band_id".

                With the pivot format, an image can be composed of several files.

                A full configuration example for pivot sensor loader is given below. In this case, multiple files are used for the same image : The file `img1.tif` refers to a panchromatic image 
                and the file `color1.tif` refers to a RGB (or RGBN) image with the same size and resolution than `img1.tif`

                .. include-cars-config:: ../../example_configs/configuration/image_pivot_loader_config


            .. tab:: Geomodel

                In most cases you do not need to fill this parameter because the RPC information can be found by CARS directly either in the image metadata or in a .XML or .RPB file.
                
                If RPC information are not in the image but in a separate file not recognized by rasterio like a .geom file, this parameter has to be filled with the path of this file.
                
                If you want to use grid models, you have to use a dictionary for the geomodel parameter and fill tge `model_type` key.

                +----------------+-----------------------+--------+---------------+------------------+----------+
                | Name           | Description           | Type   | Default value | Available values | Required |
                +================+=======================+========+===============+==================+==========+
                | *path*         | File path             | str    |               |                  | Yes      |
                +----------------+-----------------------+--------+---------------+------------------+----------+
                | *model_type*   | Geomodel type         | str    | RPC           | RPC, GRID        | No       |
                +----------------+-----------------------+--------+---------------+------------------+----------+


                .. note::
                     If the geomodel file is not provided, CARS will try to use the RPC loaded with rasterio opening *image*. RPCs are assumed to convert rows and columns into WGS84 longitude/latitude coordinates.
                     
                A full configuration example is given below : 

                .. include-cars-config:: ../../example_configs/configuration/geomodel_full_config


            .. tab:: Mask

                The mask parameter is optional. A mask can be used if you want to define an area that CARS will not process.
                
                The mask must be a mono-band binary image. Please, see the section :ref:`convert_image_to_binary_image` to make binary *mask* image with 1 bit per band.
                
                The file path must be given directly as a string parameter.

                A configuration example is given below : 

                .. include-cars-config:: ../../example_configs/configuration/mask_full_config


            .. tab:: Classification

                The classification parameter is optional. It can be used to define areas that has to be filled (particularly water and cloud).
                
                The classification must be a mono-band uint8 image.
                
                If the file path is given without other parameters, CARS will not perform any filling.

                As the image parameter, the classification parameter can use sensor loaders : 

                **Basic loader**

                If you want to define a filling method for each value, you can use the following dictionary for this parameter :

                +-----------------+--------------------------------------------------------------------+--------+--------------------------+------------------+----------+
                | Name            | Description                                                        | Type   | Default value            | Available values | Required |
                +=================+====================================================================+========+==========================+==================+==========+
                | *loader*        | Name of sensor loader                                              | str    | "basic_classif"          | "basic_classif"  | No       |
                +-----------------+--------------------------------------------------------------------+--------+--------------------------+------------------+----------+
                | *path*          | File path                                                          | str    |                          |                  | Yes      |
                +-----------------+--------------------------------------------------------------------+--------+--------------------------+------------------+----------+
                | *filling*       | Values of the classification corresponding to each filling method  | dict   | Given by the table below |                  | No       |
                +-----------------+--------------------------------------------------------------------+--------+--------------------------+------------------+----------+

                And fill the *filling* parameter as follows : 

                +----------------------------+---------------------------------------------------------------------------------+-----------+--------------------------+----------+
                | Name                       | Description                                                                     | Type      | Default value            | Required |
                +============================+=================================================================================+===========+==========================+==========+
                | *fill_with_geoid*          | Value(s) for which pixels will be filled with geoid (sea)                       | int, list | None                     | No       |
                +----------------------------+---------------------------------------------------------------------------------+-----------+--------------------------+----------+
                | *interpolate_from_borders* | Value(s) for which pixels will be filled with the value on borders (lakes)      | int, list | None                     | No       |
                +----------------------------+---------------------------------------------------------------------------------+-----------+--------------------------+----------+
                | *fill_with_endogenous_dtm* | Value(s) for which pixels will be filled with a DTM generated by CARS (rivers)  | int, list | None                     | No       |
                +----------------------------+---------------------------------------------------------------------------------+-----------+--------------------------+----------+
                | *fill_with_exogenous_dtm*  | Value(s) for which pixels will be filled with the DTM given by the user (cloud) | int, list | None                     | No       |
                +----------------------------+---------------------------------------------------------------------------------+-----------+--------------------------+----------+

                .. warning::

                    The value 0 cannot be used as a value to fill because pixels labeled 0 in classification are considered as unclassified pixels.

                For each filling method, if you fill the parameter with `none` or [], the corresponding method will not be used.

                A full configuration example is given below : 

                .. include-cars-config:: ../../example_configs/configuration/classif_basic_loader_config

                **SLURP loader**

                The SLURP loader is useful if the classification used comes from `SLURP tool <https://github.com/CNES/slurp>`_
                The loader automatically fills the *filling* dictionary according to the SLURP convention. It follows this table : 

                +-----------------+----------------------------+---------------------------+
                | Value           | Class                      | Filling method            |
                +=================+============================+===========================+
                | 8               | Sea                        | fill_with_geoid           |
                +-----------------+----------------------------+---------------------------+
                | 9               | Lake                       | interpolate_from_borders  |
                +-----------------+----------------------------+---------------------------+
                | 10              | River                      | fill_with_endogenous_dtm  |
                +-----------------+----------------------------+---------------------------+
                | 6               | Cloud                      | fill_with_exogenous_dtm   |
                +-----------------+----------------------------+---------------------------+

                To use the SLURP sensor loader, simply add a *loader* parameter with the key "slurp_classif" :

                +-----------------+--------------------------------------------------------------------+--------+--------------------------+------------------+----------+
                | Name            | Description                                                        | Type   | Default value            | Available values | Required |
                +=================+====================================================================+========+==========================+==================+==========+
                | *loader*        | Name of sensor loader                                              | str    | "basic_classif"          | "slurp_classif"  | Yes      |
                +-----------------+--------------------------------------------------------------------+--------+--------------------------+------------------+----------+
                | *path*          | File path                                                          | str    |                          |                  | Yes      |
                +-----------------+--------------------------------------------------------------------+--------+--------------------------+------------------+----------+

                For example :

                .. include-cars-config:: ../../example_configs/configuration/classif_slurp_loader_config

                **Pivot loader**

                The pivot loader is the full parametrization of the classification. It can be used to optimize the reading of classification file.

                The pivot loader looks like the basic loader but with the *values* parameter added : 

                +-----------------+--------------------------------------------------------------------+--------+--------------------------+------------------+----------+
                | Name            | Description                                                        | Type   | Default value            | Available values | Required |
                +=================+====================================================================+========+==========================+==================+==========+
                | *loader*        | Name of sensor loader                                              | str    | "basic_classif"          | "pivot_classif"  | No       |
                +-----------------+--------------------------------------------------------------------+--------+--------------------------+------------------+----------+
                | *path*          | File path                                                          | str    |                          |                  | Yes      |
                +-----------------+--------------------------------------------------------------------+--------+--------------------------+------------------+----------+
                | *filling*       | Values of the classification corresponding to each filling method  | dict   | Same as basic loader     |                  | No       |
                +-----------------+--------------------------------------------------------------------+--------+--------------------------+------------------+----------+
                | *values*        | List of values read in the classification file                     | list   | []                       |                  | Yes      |
                +-----------------+--------------------------------------------------------------------+--------+--------------------------+------------------+----------+

                With the basic loader, classes are automatically defined from statistics of the input file. But with the pivot loader, the classes must be given in the *values* parameter.

                An example is given below : 

                .. include-cars-config:: ../../example_configs/configuration/classif_pivot_loader_config


    .. tab:: Pairing

        The `pairing` attribute defines the pairs to use, using sensors keys used to define sensor images.

        .. include-cars-config:: ../../example_configs/configuration/inputs_sensor_image_pairing

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

        .. include-cars-config:: ../../example_configs/configuration/inputs_initial_elevation_1

        Alternatively, it can be set as a string corresponding to the DEM path, in which case default values for the geoid and the default altitude are used.

        .. include-cars-config:: ../../example_configs/configuration/inputs_initial_elevation_2

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

        .. include-cars-config:: ../../example_configs/configuration/inputs_roi_1

        If the *debug_with_roi* advanced parameter (see dedicated tab) is enabled, the tiling of the entire image is kept but only the tiles intersecting
        the ROI are computed.

        MultiPolygon feature is only useful if the parameter *debug_with_roi* is activated, otherwise the total footprint of the
        MultiPolygon will be used as ROI.

        By default epsg 4326 is used. If the user has defined a polygon in a different reference system, the "crs" field must be specified.

        Example of the *debug_with_roi* mode utilizing an "roi" parameter of type MultiPolygon as a feature and a specific EPSG.

        .. include-cars-config:: ../../example_configs/configuration/inputs_roi_2

        Example of the "roi" parameter with a Shapefile

        .. include-cars-config:: ../../example_configs/configuration/inputs_roi_3
