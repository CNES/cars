Output directory content
------------------------

The output directory, defined in the configuration file, contains at the end of the computation:

* the required product levels (`depth_map`, `dsm` and/or `point_cloud`)
* the dump directory (`dump_dir`) containing intermediate data for all applications
* the intermediate resolutions directory (`intermediate_res`) containing the results (and `dump_dir`) of all intermediate resolutions
* metadata json file (`metadata.json`) containing: used parameters, information and numerical results related to computation, step by step and pair by pair.
* logs folder (`logs`) containing CARS log and profiling information

.. note:: 
    DSMs and point cloud are referenced to the local UTM zone with WGS4 ellipsoid as datum (but you can use another epsg code if you need to {"output": {"epsg": MY_EPSG_CODE}}).

.. tabs::

    .. tab:: DSM

        If product type `dsm` is selected, a directory named `dsm` will be created with the DSM and every auxiliary product selected. The file `dsm/index.json` shows the path of every generated file. For example :

        .. include-cars-config:: ../../example_configs/configuration/output_dsm_output

        .. note::
            If `performance_map_method` in dense matching configuration is a list with more than one element, `performance_map.tif` will be a 3 dimension raster: each band contains the performance map for each method.
            Else, it will be a two dimension raster

    .. tab:: Depth map

        If product type `depth_map` is selected, a directory named `depth_map` will be created with a subfolder for every pair. The file `depth_map/index.json` shows the path of every generated file. For example :

        .. include-cars-config:: ../../example_configs/configuration/output_depth_map_output

        .. note::
            If `performance_map_method` in dense matching configuration is a list with more than one element, `performance_map_from_risk.tif` and `performance_map_from_intervals.tif` will be generated. Choose one to re enter with.


    .. tab:: Point cloud

        If product type `point_cloud` is selected, a directory named `point_cloud` will be created with a subfolder for every pair.

        The point cloud output product consists of a collection of laz files, each containing a tile of the point cloud.

        The point cloud found in the product the highest level point cloud produced by CARS. For exemple, if outlier removal and point cloud denoising are deactivated, the point cloud will correspond to the output of triangulation. If only the first application of outlier removal is activated, this will be the output point cloud.

         .. note::
               The resolution has no effect on the point cloud. The point cloud contains the positions calculated for each point in epipolar geometry (approximately at full sensor resolution).

        The file `point_cloud/index.json` shows the path of every generated file. For example :

        .. include-cars-config:: ../../example_configs/configuration/output_point_cloud_output
