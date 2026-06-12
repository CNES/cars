Output configuration
--------------------

+-------------------------+-------------------------------------------------------------+--------------------+------------------------+----------+
| Name                    | Description                                                 | Type               | Default value          | Required |
+=========================+=============================================================+====================+========================+==========+
| *directory*             | Output folder where results are stored                      | string             | No                     | Yes      |
+-------------------------+-------------------------------------------------------------+--------------------+------------------------+----------+
| *product_level*         | Output requested products (dsm, point_cloud, dtm)           | list or string     | "dsm"                  | No       |
+-------------------------+-------------------------------------------------------------+--------------------+------------------------+----------+
| *product_format*        | Format of the point cloud (tif, laz)                        | dict               | {"point_cloud": "laz"} | No       |
+-------------------------+-------------------------------------------------------------+--------------------+------------------------+----------+
| *resolution* [#scaled]_ | Output DSM grid step (only for dsm product level)           | float              | None [#scaled]_        | No       |
+-------------------------+-------------------------------------------------------------+--------------------+------------------------+----------+
| *auxiliary*             | Selection of additional files in products                   | dict               | See below              | No       |
+-------------------------+-------------------------------------------------------------+--------------------+------------------------+----------+
| *epsg*                  | EPSG code                                                   | int, string        | None                   | No       |
+-------------------------+-------------------------------------------------------------+--------------------+------------------------+----------+
| *geoid*                 | Output geoid                                                | bool or string     | True                   | No       |
+-------------------------+-------------------------------------------------------------+--------------------+------------------------+----------+
| *save_by_pair*          | Save output point clouds by pair                            | bool               | False                  | No       |
+-------------------------+-------------------------------------------------------------+--------------------+------------------------+----------+

.. include-cars-config:: ../../example_configs/configuration/output_1

.. tabs::

    .. tab:: Product level

        The `product_level` attribute defines which product should be produced by CARS. There are two available product type: `point_cloud` and `dsm`.

        The point cloud output format can be specified using the product_format variable. Two options are available: tif and laz.

        If `dtm` is requested in `product_level`, the DSM will also be computed.

        A single product can be requested by setting the parameter as string or several products can be requested by providing a list.

        .. tabs::

            .. tab:: DSM

                This is the default behavior of CARS : a single DSM will be generated from one or several pairs of images.

                The smallest configuration can simply contain those inputs.

                .. include-cars-config:: ../../example_configs/configuration/output_n_pairs_1_dsm

            .. tab:: Point clouds

                The point_cloud product can be generated in two formats depending on the value of product_format["point_cloud"].

                **GeoTIFF format ("tif")**

                When the format is set to "tif", CARS exports the point cloud as a set of raster layers organized on the epipolar grid. The 3D coordinates of each point are stored as raster images (X, Y and Z), preserving the geometry and dimensions of the epipolar pair from which they were generated.

                To generate these rasterized point clouds, add point_cloud to product_level and set product_format["point_cloud"] to "tif".

                .. include-cars-config:: ../../example_configs/configuration/output_n_pairs_n_depth_maps

                **LAZ format ("laz")**

                When the format is set to "laz", CARS exports the point cloud as a standard point cloud file. Each point is stored directly as a 3D point and is no longer organized on the epipolar grid.

                To generate LAZ point clouds, add ``point_cloud`` to product_level and set product_format["point_cloud"] to "laz".

                .. include-cars-config:: ../../example_configs/configuration/output_n_pairs_n_point_clouds

   
    .. tab:: Auxiliary data

        **BASIC USAGE**

        Additional auxiliary files can be produced by setting the `auxiliary` dictionary attribute.

        +-----------------------+-------------------------------------------------------------+------------------+----------------+-----------+
        | Name                  | Description                                                 | Type             | Default value  | Required  |
        +=======================+=============================================================+==================+================+===========+
        | *image*               | Save output orthorectified image                            | bool, str, list  | True           | No        |
        +-----------------------+-------------------------------------------------------------+------------------+----------------+-----------+
        | *classification*      | Save output classification map                              | bool, dict, list | False          | No        |
        +-----------------------+-------------------------------------------------------------+------------------+----------------+-----------+
        | *filling*             | Save output filling                                         | bool, dict       | False          | No        |
        +-----------------------+-------------------------------------------------------------+------------------+----------------+-----------+
        | *performance_map*     | Save output performance map                                 | bool, list       | False          | No        |
        +-----------------------+-------------------------------------------------------------+------------------+----------------+-----------+
        | *weights*             | Save output dsm weights                                     | bool             | False          | No        |
        +-----------------------+-------------------------------------------------------------+------------------+----------------+-----------+
        | *contributing_pair*   | Save output contributing pair                               | bool             | False          | No        |
        +-----------------------+-------------------------------------------------------------+------------------+----------------+-----------+
        | *ambiguity*           | Save output ambiguity                                       | bool             | False          | No        |
        +-----------------------+-------------------------------------------------------------+------------------+----------------+-----------+
        | *edges*               | Save output edge rasters (depth_map only)                   | bool             | False          | No        |
        +-----------------------+-------------------------------------------------------------+------------------+----------------+-----------+

        In the table below, the default value corresponds to the value used if parameter is set to `True`. For the classification, the output values will correspond to the input ones.

        +-----------------------+----------------------------------------------------------------------+--------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------+
        | Name                  | Description                                                          | Type   | Default value                                                                                                                                                                      | Required  |
        +=======================+======================================================================+========+====================================================================================================================================================================================+===========+
        | *image*               | Define the order of the bands on the output image                    | list   | [b0, b1, b2]  (phr images), [b1, b0, b2] (co3d images)                                                                                                                             | No        |
        +-----------------------+----------------------------------------------------------------------+--------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------+
        | *classification*      | Edit and/or merge the values of the classification map               | dict   | {n:n} with n the label of the classification                                                                                                                                       | No        |
        +-----------------------+----------------------------------------------------------------------+--------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------+
        | *filling*             | Edit and/or merge the values of the filling map                      | dict   | {0: "no_data", 1: "no_edition", 2: "fill_with_exogenous_dem", 3: "interpolation", 4: "fill_with_endogenous_dem", 5: "interpolate_from_borders", 6: "fill_with_geoid", 7: "other"}  | No        |
        +-----------------------+----------------------------------------------------------------------+--------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------+
        | *performance_map*     | List defining intervals used in the performance map classification   | list   | [0, 0.968, 1.13375, 1.295, 1.604, 2.423, 3.428]                                                                                                                                    | No        |
        +-----------------------+----------------------------------------------------------------------+--------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------+

        .. include-cars-config:: ../../example_configs/configuration/output_auxiliary_basic

        Note that not all rasters associated to the DSM that CARS can produce are available in the output product auxiliary data. For example, confidence intervals are not part of the output product but can be found in the rasterization `dump_dir` if `save_intermediate_data` is activated in the `rasterization` application configuration.

        .. note::

            The auxiliary parameter ``edges`` currently acts at triangulation level.
            When enabled, it exports edge data in the depth map product (``depth_map`` in ``product_level``).
            It currently does not work with DSM auxiliary outputs, and won't save additional data for the DSM.

        **ADVANCED USAGE**
        
        For some auxiliary data, output options can be configured inside this dictionary by overloading the boolean parameter. 
        
        For each auxiliary file, if a non-boolean parameter is given, the file is saved according to this parameter.

        .. include-cars-config:: ../../example_configs/configuration/output_auxiliary_advanced

        **Image parameter**:

        For this specific configuration (see yaml file above), only one band (b1) is written in the output image.

        **Classification parameter**:

        For this specific configuration (see yaml file above), the output classification takes the following values:

        +-----------------------+----------------+
        | input values          | output values  |
        +=======================+================+
        | [1, 2]                | 17             |
        +-----------------------+----------------+
        | 3                     | 3              |
        +-----------------------+----------------+
        | 4                     | 15             |
        +-----------------------+----------------+

        **Filling parameter**:

        For this specific configuration (see yaml file above), the output filling takes the following values:

        +--------------------------+----------------+
        | input values             | output values  |
        +==========================+================+
        | fill_with_geoid          | 18             |
        +--------------------------+----------------+
        | interpolate_from_border  | 18             |
        +--------------------------+----------------+
        | fill_with_endogenous_dem | 3              |
        +--------------------------+----------------+
        | interpolation            | 9              |
        +--------------------------+----------------+
        | other filling methods    | default values |
        +--------------------------+----------------+

        **Performance map parameter**:

        For this specific configuration (see yaml file above), those values will be taken into account in the output file. You can modify the list length according to your needs.

    .. tab:: EPSG

        This parameter defines the EPSG code to which the output data will be referenced.
        If set to None, CARS will automatically use the EPSG code of the most suitable UTM zone for the input data.

        .. include-cars-config:: ../../example_configs/configuration/output_epsg_1

        When combined with the Geoid parameter, the EPSG ensures that the output file is assigned a CRS that also includes the corresponding vertical reference system.
        
        .. include-cars-config:: ../../example_configs/configuration/output_epsg_2

        Additionally, this parameter can be used to override the vertical CRS of the output data, by specifying either a 3D CRS or a CompoundCRS.
        For example, if the geoid provided is associated with a specific EPSG code that CARS cannot automatically detect, you can explicitly set it here.

        .. include-cars-config:: ../../example_configs/configuration/output_epsg_3                


    .. tab:: Geoid

        This parameter refers to the vertical reference of the output product, used as an altitude offset during triangulation.
        It can be set as a string to provide the path to a geoid file on disk, or as a boolean: if set to `True` CARS default geoid is used,
        if set to `False` no vertical offset is applied (ellipsoid reference).

        If the EPSG parameter does not already define a vertical reference, a Vertical CRS (VCRS) is derived from the `Geoid` parameter.

        - If set to ``False``, a WKT corresponding to WGS84 is used.
        - If set to ``True``, the default EGM96 model (EPSG:5773) is used.
        - If set to a file path, the geoid file name is used to determine the appropriate VCRS. Currently, only EGM96 and EGM08 are supported.

        If the provided file is not recognized, a WKT referencing the file directly is created instead.

.. rubric:: Footnotes

.. [#scaled] This parameter is computed at runtime depending on the resolution of the input sensor images. You can still override it in the configuration.
