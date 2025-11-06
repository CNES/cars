Output configuration
--------------------

+-------------------------+-------------------------------------------------------------+--------------------+-----------------------+----------+
| Name                    | Description                                                 | Type               | Default value         | Required |
+=========================+=============================================================+====================+=======================+==========+
| *directory*             | Output folder where results are stored                      | string             | No                    | Yes      |
+-------------------------+-------------------------------------------------------------+--------------------+-----------------------+----------+
| *product_level*         | Output requested products (dsm, point_cloud, depth_map)     | list or string     | "dsm"                 | No       |
+-------------------------+-------------------------------------------------------------+--------------------+-----------------------+----------+
| *resolution* [#scaled]_ | Output DSM grid step (only for dsm product level)           | float              | None [#scaled]_       | No       |
+-------------------------+-------------------------------------------------------------+--------------------+-----------------------+----------+
| *auxiliary*             | Selection of additional files in products                   | dict               | See below             | No       |
+-------------------------+-------------------------------------------------------------+--------------------+-----------------------+----------+
| *epsg*                  | EPSG code                                                   | int, string        | None                  | No       |
+-------------------------+-------------------------------------------------------------+--------------------+-----------------------+----------+
| *geoid*                 | Output geoid                                                | bool or string     | True                  | No       |
+-------------------------+-------------------------------------------------------------+--------------------+-----------------------+----------+
| *save_by_pair*          | Save output point clouds by pair                            | bool               | False                 | No       |
+-------------------------+-------------------------------------------------------------+--------------------+-----------------------+----------+

.. include-cars-config:: ../../example_configs/configuration/output_1

.. tabs::

    .. tab:: Product level

        The `product_level` attribute defines which product should be produced by CARS. There are three available product type: `depth_map`, `point_cloud` and `dsm`.

        A single product can be requested by setting the parameter as string or several products can be requested by providing a list.

        .. tabs::

            .. tab:: DSM

                This is the default behavior of CARS : a single DSM will be generated from one or several pairs of images.

                The smallest configuration can simply contain those inputs.

                .. include-cars-config:: ../../example_configs/configuration/output_n_pairs_1_dsm

            .. tab:: Depth Maps

                Depth maps are a way to represent point clouds as three images X Y and Z, each one representing the position of a pixel on its axis.
                They are an official product of CARS.

                The ``product_level`` key in ``output`` can contain any combination of the values `dsm`, `depth_map`, and `point_cloud`.

                Depth maps (one for each sensor pair) will be saved if `depth_map` is present in ``product_level`` :

                .. include-cars-config:: ../../example_configs/configuration/output_n_pairs_n_depth_maps

            .. tab:: Point clouds

                Just like depth maps, the point cloud is an official product of CARS. As such, all that's needed is to add `point_cloud` to ``product_level`` in order for it to be generated.

                .. include-cars-config:: ../../example_configs/configuration/output_n_pairs_n_point_clouds

   
    .. tab:: Auxiliary data

        **Basic usage**

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

        .. include-cars-config:: ../../example_configs/configuration/output_auxiliary_basic

        Note that not all rasters associated to the DSM that CARS can produce are available in the output product auxiliary data. For example, confidence intervals are not part of the output product but can be found in the rasterization `dump_dir` if `save_intermediate_data` is activated in the `rasterization` application configuration.

        **Advanced usage**
        
        For some auxiliary data, output options can be configured inside this dictionary by overloading the boolean parameter. 
        
        For each auxiliary file, if a non-boolean parameter is given, the file is saved according to this parameter.

        In the table below, the default value corresponds to the value used if parameter is set to `True`.

        +-----------------------+----------------------------------------------------------------------+--------+---------------------------------------------------------------------------------------------------------------------+-----------+
        | Name                  | Description                                                          | Type   | Default value                                                                                                       | Required  |
        +=======================+======================================================================+========+=====================================================================================================================+===========+
        | *image*               | Define the order of the bands on the output image                    | list   | [b0, b1, b2, ...]                                                                                                   | No        |
        +-----------------------+----------------------------------------------------------------------+--------+---------------------------------------------------------------------------------------------------------------------+-----------+
        | *classification*      | Edit and/or merge the values of the classification map               | dict   | {1: 1, 2: 2, ...}                                                                                                   | No        |
        +-----------------------+----------------------------------------------------------------------+--------+---------------------------------------------------------------------------------------------------------------------+-----------+
        | *filling*             | Edit and/or merge the values of the filling map                      | dict   | {1: "fill_with_geoid", 2: "interpolate_from_borders", 3: "fill_with_endogenous_dem", 4: "fill_with_exogenous_dem"}  | No        |
        +-----------------------+----------------------------------------------------------------------+--------+---------------------------------------------------------------------------------------------------------------------+-----------+
        | *performance_map*     | List defining intervals used in the performance map classification   | list   | [0, 0.968, 1.13375, 1.295, 1.604, 2.423, 3.428]                                                                     | No        |
        +-----------------------+----------------------------------------------------------------------+--------+---------------------------------------------------------------------------------------------------------------------+-----------+

        .. note::

           For the image parameter:
            You can also put a string instead of a list or a boolean. Therefore, only one band will be used for the color.


           For the classification parameter:
            you can configure the output. For example, by configuring `{17: [1, 2], 3:3, 15:4}`:

            - The classification represented by the pixel value 1 and 2 in the input file will be represented by the value 17 in the output file.
            - The classification represented by the pixel value 3 in the input file will be represented by the value 3 in the output file.
            - The classification represented by the pixel value 4 in the input file will be represented by the value 15 in the output file.
           
            Here, if a pixel is classified by the values 1 and 3, the priority will go to the first value in the dictionary -> 17.
           
            You can also use a list as [3, 2, 1, 4] and a dictionary will be formatted regarding those values: {3: 1, 2: 2, 1: 3, 4: 4}. By using a list, it will not be possible for you to use values that are not in the classification input file (as 17 in this example).

           It works the same for the filling parameter:
            By configuring `{17: ["fill_with_geoid", "interpolate_from_borders"], 3: "fill_with_endogenous_dem", 15: "fill_with_exogenous_dem", "32": "other"}`:

            - Pixels filled with the `fill_with_geoid` and `interpolate_from_borders` methods will be represented by the value 17 in the output file.
            - Pixels filled with the `fill_with_endogenous_dem` method will be represented by the value 3 in the output file.
            - Pixels filled with the `fill_with_exogenous_dem` method will be represented by the value 15 in the output file.
            - Pixels filled with other methods will be represented by the value 32 in the output file. By default, the value is 50.

        .. include-cars-config:: ../../example_configs/configuration/output_auxiliary_advanced

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
