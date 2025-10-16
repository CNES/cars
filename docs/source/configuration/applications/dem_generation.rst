DEM Generation
==============

**Name**: "dem_generation"

**Description**

Generates dem from sparse matches, and fits the initial elevation onto the median dem.

Up to 4 dems are generated, with different methods:

* median
* min
* max
* initial_elevation_fit (only if ``coregistration`` is set to ``true``)

The DEMs are generated in the application dump directory.
You can find the shift values applied to the initial elevation in ``metadata.json``.

**Configuration**

+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------+----------+
| Name                            | Description                                                              | Type       | Available value                      | Default value          | Required |
+=================================+==========================================================================+============+======================================+========================+==========+
| method                          | Method for dem_generation                                                | string     | "dichotomic", "bulldozer_on_raster"  | "bulldozer_on_raster"  | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------+----------+
| height_margin [#scaled]_        | Height margin [margin min, margin max], in meter                         | int        |                                      | 5 [#scaled]_           | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------+----------+
| min_dem                         | Min value that has to be reached by dem_min                              | int        | should be < 0                        | -500                   | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------+----------+
| max_dem                         | Max value that has to be reached by dem_max                              | int        | should be > 0                        | 1000                   | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------+----------+
| coregistration                  | Use the median dem to correct shifts in the initial elevation provided   | boolean    |                                      | true                   | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------+----------+
| coregistration_max_shift        | Maximum shift allowed on X/Y axes for the coregistered initial elevation | int, float | should be > 0                        | 180                    | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------+----------+
| save_intermediate_data          | Save DEM as TIF                                                          | boolean    |                                      | false                  | No       |
+---------------------------------+--------------------------------------------------------------------------+------------+--------------------------------------+------------------------+----------+

**Method dichotomic**

Generates DEM min and max from percentiles of matches altitude grouped by cells of a regular grid

+---------------------------------+----------------------------------------------------------------------------+------------+-----------------+-----------------+----------+
| Name                            | Description                                                                | Type       | Available value | Default value   | Required |
+=================================+============================================================================+============+=================+=================+==========+
| resolution                      | Resolution of dem, in meter                                                | int, float | should be > 0   | 90              | No       |
+---------------------------------+----------------------------------------------------------------------------+------------+-----------------+-----------------+----------+
| margin                          | Margin to use on the border of dem: [factor_of_dem_size, margin_in_meters] | list       | should be > 0   | [0, 6000]       | No       |
+---------------------------------+----------------------------------------------------------------------------+------------+-----------------+-----------------+----------+
| fillnodata_max_search_distance  | Max search distance for rasterio fill nodata                               | int        | should be > 0   | 3               | No       |
+---------------------------------+----------------------------------------------------------------------------+------------+-----------------+-----------------+----------+
| percentile                      | Percentile of matches to ignore in min and max functions                   | int        | should be > 0   | 1               | No       |
+---------------------------------+----------------------------------------------------------------------------+------------+-----------------+-----------------+----------+
| min_number_matches              | Minimum number of matches needed to have a valid tile                      | int        | should be > 0   | 30              | No       |
+---------------------------------+----------------------------------------------------------------------------+------------+-----------------+-----------------+----------+

**Method bulldozer_on_raster**

Rasterizes all matches on a regular grid and performs morphological operations and Bulldozer processing to compute DEM min and max

+-------------------------------------+---------------------------------------------------------------------------------+------------+-----------------+-------------------------+----------+
| Name                                | Description                                                                     | Type       | Available value | Default value           | Required |
+=====================================+=================================================================================+============+=================+=========================+==========+
| margin [#scaled]_                   | Margin to use on the border of dem: [factor_of_dem_size, margin_in_meters]      | list       | should be > 0   | [0.2, None [#scaled]_ ] | No       |
+-------------------------------------+---------------------------------------------------------------------------------+------------+-----------------+-------------------------+----------+
| fillnodata_max_search_distance      | Max search distance for rasterio fill nodata                                    | int        | should be > 0   | 50                      | No       |
+-------------------------------------+---------------------------------------------------------------------------------+------------+-----------------+-------------------------+----------+
| morphological_filters_size          | Size (in pixels) of erosion and dilation filters used to generate DEM           | int        | should be > 0   | 30                      | No       |
+-------------------------------------+---------------------------------------------------------------------------------+------------+-----------------+-------------------------+----------+
| preprocessing_median_filter_size    | Size (in pixels) of first median filter used to smooth median DEM               | int        | should be > 0   | 5                       | No       |
+-------------------------------------+---------------------------------------------------------------------------------+------------+-----------------+-------------------------+----------+
| dem_median_downscale                | Downsample factor on dsm to generate median DEM                                 | int        | should be > 0   | 10                      | No       |
+-------------------------------------+---------------------------------------------------------------------------------+------------+-----------------+-------------------------+----------+
| dem_min_max_downscale               | Downsample factor on dsm to generate DEM min and DEM max                        | int        | should be > 0   | 10                      | No       |
+-------------------------------------+---------------------------------------------------------------------------------+------------+-----------------+-------------------------+----------+
| postprocessing_median_filter_size   | Size (in pixels) of second median filter used to smooth downsampled median DEM  | int        | should be > 0   | 7                       | No       |
+-------------------------------------+---------------------------------------------------------------------------------+------------+-----------------+-------------------------+----------+
| bulldozer_max_object_size           | Bulldozer parameter "max_object_size"                                           | int        | should be > 0   | 16                      | No       |
+-------------------------------------+---------------------------------------------------------------------------------+------------+-----------------+-------------------------+----------+
| compute_stats                       | Compute statistics of difference between DEM min/max and original DSM           | boolean    |                 | true                    | No       |
+-------------------------------------+---------------------------------------------------------------------------------+------------+-----------------+-------------------------+----------+
| disable_bulldozer                   | Disable Bulldozer step to fasten the DEM generation                             | boolean    |                 | false                   | No       |
+-------------------------------------+---------------------------------------------------------------------------------+------------+-----------------+-------------------------+----------+

**Example**

.. include-cars-config:: ../../example_configs/configuration/applications_dem_generation

.. rubric:: Footnotes

.. [#scaled] This parameter is computed at runtime depending on the resolution of the input sensor images. You can still override it in the configuration.

