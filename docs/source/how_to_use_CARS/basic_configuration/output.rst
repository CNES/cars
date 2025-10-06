.. _output:

Output
======

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

.. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/output_1

.. tabs::

    .. tab:: Product level

        The `product_level` attribute defines which product should be produced by CARS. There are three available product type: `depth_map`, `point_cloud` and `dsm`.

        A single product can be requested by setting the parameter as string or several products can be requested by providing a list.

        .. tabs::

            .. tab:: DSM

                This is the default behavior of CARS : a single DSM will be generated from one or several pairs of images.

                The smallest configuration can simply contain those inputs.

                .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/output_n_pairs_1_dsm

            .. tab:: Depth Maps

                Depth maps are a way to represent point clouds as three images X Y and Z, each one representing the position of a pixel on its axis.
                They are an official product of CARS.

                The ``product_level`` key in ``output`` can contain any combination of the values `dsm`, `depth_map`, and `point_cloud`.

                Depth maps (one for each sensor pair) will be saved if `depth_map` is present in ``product_level`` :

                .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/output_n_pairs_n_depth_maps

            .. tab:: Point clouds

                Just like depth maps, the point cloud is an official product of CARS. As such, all that's needed is to add `point_cloud` to ``product_level`` in order for it to be generated.

                .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/output_n_pairs_n_point_clouds

   
    .. tab:: Auxiliary data

        **Basic usage**

        Additional auxiliary files can be produced by setting the `auxiliary` dictionary attribute.

        +-----------------------+-------------------------------------------------------------+--------+----------------+-----------+
        | Name                  | Description                                                 | Type   | Default value  | Required  |
        +=======================+=============================================================+========+================+===========+
        | *image*               | Save output orthorectified image                            | bool   | True           | No        |
        +-----------------------+-------------------------------------------------------------+--------+----------------+-----------+
        | *classification*      | Save output classification map                              | bool   | False          | No        |
        +-----------------------+-------------------------------------------------------------+--------+----------------+-----------+
        | *filling*             | Save output filling                                         | bool   | False          | No        |
        +-----------------------+-------------------------------------------------------------+--------+----------------+-----------+
        | *performance_map*     | Save output performance map                                 | bool   | False          | No        |
        +-----------------------+-------------------------------------------------------------+--------+----------------+-----------+
        | *weights*             | Save output dsm weights                                     | bool   | False          | No        |
        +-----------------------+-------------------------------------------------------------+--------+----------------+-----------+
        | *contributing_pair*   | Save output contributing pair                               | bool   | False          | No        |
        +-----------------------+-------------------------------------------------------------+--------+----------------+-----------+
        | *ambiguity*           | Save output ambiguity                                       | bool   | False          | No        |
        +-----------------------+-------------------------------------------------------------+--------+----------------+-----------+

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/output_auxiliary_basic

        Note that not all rasters associated to the DSM that CARS can produce are available in the output product auxiliary data. For example, confidence intervals are not part of the output product but can be found in the rasterization `dump_dir` if `save_intermediate_data` is activated in the `rasterization` application configuration.

        **Advanced usage**
        
        For some auxiliary data, output options can be configured inside this dictionary by overloading the boolean parameter. 
        
        For each auxiliary file, if a non-boolean parameter is given, the file is saved according to this parameter.

        In the table below, the default value corresponds to the value used if parameter is set to `True`.

        +-----------------------+----------------------------------------------------------------------+--------+-------------------------------------------------+-----------+
        | Name                  | Description                                                          | Type   | Default value                                   | Required  |
        +=======================+======================================================================+========+=================================================+===========+
        | *image*               | Define the order of the bands on the output image                    | list   | [b0, b1, b2, ...]                               | No        |
        +-----------------------+----------------------------------------------------------------------+--------+-------------------------------------------------+-----------+
        | *classification*      | Edit and/or merge the values of the classification map               | dict   | {1: 1, 2: 2, ...}                               | No        |
        +-----------------------+----------------------------------------------------------------------+--------+-------------------------------------------------+-----------+
        | *filling*             | Edit and/or merge the values of the filling map                      | dict   | {1: 1, 2: 2, ...}                               | No        |
        +-----------------------+----------------------------------------------------------------------+--------+-------------------------------------------------+-----------+
        | *performance_map*     | List defining intervals used in the performance map classification   | list   | [0, 0.968, 1.13375, 1.295, 1.604, 2.423, 3.428] | No        |
        +-----------------------+----------------------------------------------------------------------+--------+-------------------------------------------------+-----------+

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/output_auxiliary_advanced

    .. tab:: EPSG

        This parameter defines the EPSG code to which the output data will be referenced.
        If set to None, CARS will automatically use the EPSG code of the most suitable UTM zone for the input data.

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/output_epsg_1

        When combined with the Geoid parameter, the EPSG ensures that the output file is assigned a CRS that also includes the corresponding vertical reference system.
        
        .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/output_epsg_2

        Additionally, this parameter can be used to override the vertical CRS of the output data, by specifying either a 3D CRS or a CompoundCRS.
        For example, if the geoid provided is associated with a specific EPSG code that CARS cannot automatically detect, you can explicitly set it here.

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/output_epsg_3                


    .. tab:: Geoid

        This parameter refers to the vertical reference of the output product, used as an altitude offset during triangulation.
        It can be set as a string to provide the path to a geoid file on disk, or as a boolean: if set to `True` CARS default geoid is used,
        if set to `False` no vertical offset is applied (ellipsoid reference).

        If the EPSG parameter does not already define a vertical reference, a Vertical CRS (VCRS) is derived from the `Geoid` parameter.

        - If set to ``False``, a WKT corresponding to WGS84 is used.
        - If set to ``True``, the default EGM96 model (EPSG:5773) is used.
        - If set to a file path, the geoid file name is used to determine the appropriate VCRS. Currently, only EGM96 and EGM08 are supported.

        If the provided file is not recognized, a WKT referencing the file directly is created instead.

**Output contents**

The output directory, defined in the configuration file, contains at the end of the computation:

* the required product levels (`depth_map`, `dsm` and/or `point_cloud`)
* the dump directory (`dump_dir`) containing intermediate data for all applications
* the intermediate resolutions directory (`intermediate_res`) containing the results (and `dump_dir`) of all intermediate resolutions
* metadata json file (`metadata.json`) containing: used parameters, information and numerical results related to computation, step by step and pair by pair.
* logs folder (`logs`) containing CARS log and profiling information

.. tabs::

    .. tab:: DSM

        If product type `dsm` is selected, a directory named `dsm` will be created with the DSM and every auxiliary product selected. The file `dsm/index.json` shows the path of every generated file. For example :

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/output_dsm_output

        .. note::
            If `performance_map_method` in dense matching configuration is a list with more than one element, `performance_map.tif` will be a 3 dimension raster: each band contains the performance map for each method.
            Else, it will be a two dimension raster

    .. tab:: Depth map

        If product type `depth_map` is selected, a directory named `depth_map` will be created with a subfolder for every pair. The file `depth_map/index.json` shows the path of every generated file. For example :

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/output_depth_map_output

        .. note::
            If `performance_map_method` in dense matching configuration is a list with more than one element, `performance_map_from_risk.tif` and `performance_map_from_intervals.tif` will be generated. Choose one to re enter with.


    .. tab:: Point cloud

        If product type `point_cloud` is selected, a directory named `point_cloud` will be created with a subfolder for every pair.

        The point cloud output product consists of a collection of laz files, each containing a tile of the point cloud.

        The point cloud found in the product the highest level point cloud produced by CARS. For exemple, if outlier removal and point cloud denoising are deactivated, the point cloud will correspond to the output of triangulation. If only the first application of outlier removal is activated, this will be the output point cloud.

        The file `point_cloud/index.json` shows the path of every generated file. For example :

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/basic_configuration/output_point_cloud_output

.. rubric:: Footnotes

.. [#scaled] This parameter is computed at runtime depending on the resolution of the input sensor images. You can still override it in the configuration.