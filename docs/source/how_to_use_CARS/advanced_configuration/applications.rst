.. _applications:

Applications
============

Tha `applications` key is optional and is used to redefine parameters for each application used in the pipeline. 
You can personnalize the configuration for each resolution at which the pipeline is ran, or override the parameters for all resolutions at once, as explained in the section right below. 

.. tabs::

    .. tab:: Overriding all resolutions at once

        This is the default behaviour when providing a configuration dict directly in the `applications` key.

        This example overrides the configuration of `application_name` for all resolutions at once :

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_override_all_resolutions


    .. tab:: Overriding a single resolution

        To override a configuration at a specific resolution, you first need to identify which resolution you want to modify. By default, CARS uses the resolutions 16, 4, and 1 :
        
    - Resolution 16 corresponds to 16 times the original resolution (e.g., 16m if the original resolution is 1m).
    - Resolution 4 corresponds to 4 times the original resolution (e.g., 4m if the original resolution is 1m).
    - Resolution 1 corresponds to the original resolution (e.g., 1m).

        Once you have chosen the resolution value, you can override the configuration by adding an entry to the `applications` dictionary with the key `resolution_{resolution_value}` with resolution value an integer.

        The following example overrides the configuration for `application_name` at resolutions 4 and 1, using different parameters for each. Resolution 16 will retain its default configuration.

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_override_single_resolution

By default, the configuration can be different for the first resolution, the intermediate resolution(s) and the last resolution. 
The changes to the default values can be modified in the source code, in ``cars/pipelines/conf_resolution/*``.

The section below includes the files directly.

.. tabs::

    .. tab:: Overriding configuration : first resolution

        This is empty for now.
        
        .. include:: ../../../../cars/pipelines/conf_resolution/conf_first_resolution.json
            :literal:

    .. tab:: Overriding configuration : all intermediate resolutions
        
        This is empty for now.

        .. include:: ../../../../cars/pipelines/conf_resolution/conf_intermediate_resolution.json
            :literal:

    .. tab:: Overriding configuration : final resolution
        
        .. include:: ../../../../cars/pipelines/conf_resolution/conf_final_resolution.json
            :literal:

The section below describes all the available parameters for each CARS application.

CARS applications are defined and called by their **name**. An example configuration is provided for each application.

Be careful with these parameters: no mechanism ensures consistency between applications for now. Some parameters can degrade performance and DSM quality heavily.
The default parameters have been set as a robust and consistent end to end configuration for the whole pipeline.

.. tabs::

    .. tab:: Grid Generation

        **Name**: "grid_generation"

        **Description**

        From sensors image, compute the stereo-rectification grids

        **Configuration**

        +-------------------------+-----------------------------------------------+---------+-----------------------------------+---------------+----------+
        | Name                    | Description                                   | Type    |     Available values              | Default value | Required |
        +=========================+===============================================+=========+===================================+===============+==========+
        | method                  | Method for grid generation                    | string  | "epipolar"                        | epipolar      | No       |
        +-------------------------+-----------------------------------------------+---------+-----------------------------------+---------------+----------+
        | epi_step                | Step of the deformation grid in nb. of pixels | int     | should be > 0                     | 30            | No       |
        +-------------------------+-----------------------------------------------+---------+-----------------------------------+---------------+----------+
        | save_intermediate_data  | Save the generated grids                      | boolean |                                   | false         | No       |
        +-------------------------+-----------------------------------------------+---------+-----------------------------------+---------------+----------+

        **Example**

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_grid_generation

    .. tab:: Resampling

        **Name**: "resampling"

        **Description**

        Input images are resampled with grids.

        **Configuration**

        +------------------------+--------------------------------------------------------+---------+-----------------+---------------+----------+
        | Name                   | Description                                            | Type    | Available value | Default value | Required |
        +========================+========================================================+=========+=================+===============+==========+
        | method                 | Method for resampling                                  | string  | "bicubic"       | "bicubic"     | No       |
        +------------------------+--------------------------------------------------------+---------+-----------------+---------------+----------+
        | strip_height           | Height of strip (only when tiling is done by strip)    | int     | should be > 0   | 60            | No       |
        +------------------------+--------------------------------------------------------+---------+-----------------+---------------+----------+
        | step                   | Horizontal step for resampling inside a strip          | int     | should be > 0   | 500           | No       |
        +------------------------+--------------------------------------------------------+---------+-----------------+---------------+----------+
        | save_intermediate_data | Save epipolar images and texture                       | boolean |                 | false         | No       |
        +------------------------+--------------------------------------------------------+---------+-----------------+---------------+----------+

        **Example**

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_resampling

    .. tab:: Sparse matching

        **Name**: "sparse_matching"

        **Description**

        Compute keypoints matches on pair images

        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | Name                                 | Description                                                                                    | Type        | Available value           | Default value | Required |
        +======================================+================================================================================================+=============+===========================+===============+==========+
        | disparity_margin                     | Add a margin to min and max disparity as percent of the disparity range.                       | float       |                           | 0.02          | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | epipolar_error_upper_bound           | Expected upper bound for epipolar error in pixels                                              | float       | should be > 0             | 10.0          | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | epipolar_error_maximum_bias          | Maximum bias for epipolar error in pixels                                                      | float       | should be >= 0            | 150.0         | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | sift_back_matching                   | Also check that right vs. left gives same match                                                | boolean     |                           | true          | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | match_filter_knn                     | Number of neighbors used to measure isolation of matches and detect isolated matches           | int         | should be > 0             | 25            | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | match_filter_constant                | Constant added to the threshold used for computing statistical outliers                        | int, float  | should be >= 0            | 0             | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | match_filter_mean_factor             | Factor of mean of isolation of matches to compute threshold of outliers                        | int, float  | should be >= 0            | 1.3           | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | match_filter_dev_factor              | Factor of deviation of isolation of matches to compute threshold of outliers                   | int, float  | should be >= 0            | 3.0           | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | save_intermediate_data               | Save matches in epipolar geometry (4 first columns) and sensor geometry (4 last columns)       | boolean     |                           | false         | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | strip_margin                         | Margin to use on strip                                                                         | int         | should be > 0             | 10            | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | elevation_delta_lower_bound          | Expected lower bound for elevation delta with respect to input low resolution dem in meters    | int, float  |                           | None          | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | elevation_delta_upper_bound          | Expected upper bound for elevation delta with respect to input low resolution dem in meters    | int, float  |                           | None          | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | minimum_nb_matches                   | Minimum number of matches that must be computed to continue pipeline                           | int         | should be > 0             | 100           | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | used_band                            | Name of band used for correlation                                                              | int         | should be in input sensor | "b0"          | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | sift_matching_threshold              | Threshold for the ratio to nearest second match                                                | float       | should be > 0             | 0.7           | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | sift_n_octave                        | The number of octaves of the Difference of Gaussians scale space                               | int         | should be > 0             | 8             | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | sift_n_scale_per_octave              | The numbers of levels per octave of the Difference of Gaussians scale space                    | int         | should be > 0             | 3             | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | sift_peak_threshold                  | Constrast threshold to discard a match (at None it will be set according to image type)        | float       | should be > 0             | 4.0           | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | sift_edge_threshold                  | Distance to image edge threshold to discard a match                                            | float       |                           | 10.0          | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | sift_magnification                   | The descriptor magnification factor                                                            | float       | should be > 0             | 7.0           | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | sift_window_size                     | smaller values let the center of the descriptor count more                                     | int         | should be > 0             | 2             | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | decimation_factor                    | Reduce the number of sifts                                                                     | int         | should be > 0             | 30            | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
        | disparity_bounds_estimation          | Parameters for the estimation of disparity interval                                            | dict        |                           | True          | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+

        For more information about these parameters, please refer to the `VLFEAT SIFT documentation <https://www.vlfeat.org/api/sift.html>`_.

        .. note::

            'elevation_delta_lower_bound' and 'elevation_delta_upper_bound' are overidden to  [-1000, 9000] in default pipeline if no initial elevation is set.
            If initial elevation is set, it is overridden to [-500, 1000].

        .. note::
            For the decimation factor, a value of 33 means that we divide the number of sift by 3, a value of 100 means that we do not decimate them


        Disparity bounds estimation:

        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
        | Name                                 | Description                                                                                    | Type        | Available value        | Default value         | Required |
        +======================================+================================================================================================+=============+========================+=======================+==========+
        | activated                            | activates estimation of disparity interval from SIFT matches                                   | bool        |                        | True                  | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
        | percentile                           | percentage of SIFT matches to ignore                                                           | int         |                        | 1                     | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
        | upper_margin                         | margin (in meters) added to altitude of higher SIFT match retained                             | int         |                        | 1000                  | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
        | lower_margin                         | margin (in meters) substracted from altitude of lower SIFT match retained                      | int         |                        | 500                   | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+


        **Example**

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_sparse_matching

    .. tab:: DEM Generation

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

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_dem_generation

    .. tab:: Ground truth reprojection

        **Name**: "ground_truth_reprojection"

        **Description**

        Generates epipolar and sensor ground truth from input dsm using direct localization.
        
        * Sensor ground truth contains altitude in sensor geometry.
        * Epipolar ground truth contains disparity map in epipolar geometry.

        +---------------------------------+------------------------------------------------------------+------------+------------------------------+---------------+----------+
        | Name                            | Description                                                | Type       | Available value              | Default value | Required |
        +=================================+============================================================+============+==============================+===============+==========+
        | method                          | Method for ground_truth_reprojection                       | string     | "direct_loc"                 |               | Yes      |
        +---------------------------------+------------------------------------------------------------+------------+------------------------------+---------------+----------+
        | target                          | Type of ground truth                                       | string     | "epipolar", "sensor", "all"  | "epipolar"    | No       |
        +---------------------------------+------------------------------------------------------------+------------+------------------------------+---------------+----------+
        | tile_size                       | Tile size to use                                           | int        |                              | 2500          | No       |
        +---------------------------------+------------------------------------------------------------+------------+------------------------------+---------------+----------+

        **Example**

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_ground_truth_reprojection

        .. figure:: ../../images/cars_pipeline_advanced.png
            :align: center
            :alt: Applications

    .. tab:: Dense matching

        **Name**: "dense_matching"

        **Description**

        Compute the disparity map from stereo-rectified pair images

        .. list-table:: Configuration
            :widths: 19 19 19 19 19 19
            :header-rows: 1

            * - Name
              - Description
              - Type
              - Available value
              - Default value
              - Required
            * - method
              - Method for dense matching
              - string
              - "census_sgm_default", "mccnn_sgm", "census_sgm_urban", "census_sgm_shadow", "census_sgm_mountain_and_vegetation", "census_sgm_homogeneous", "auto"
              - "auto"
              - No
            * - loader
              - external library use to compute dense matching
              - string
              - "pandora"
              - "pandora"
              - No
            * - loader_conf
              - Configuration associated with loader, dictionary or path to config
              - dict or str
              -
              -
              - No
            * - min_elevation_offset
              - Override minimum disparity from prepare step with this offset in meters
              - int
              -
              - None
              - No
            * - max_elevation_offset
              - Override maximum disparity from prepare step with this offset in meters
              - int
              - should be > min
              - None
              - No
            * - disp_min_threshold
              - Override minimum disparity when less than lower bound
              - int
              -
              - None
              - No
            * - disp_max_threshold
              - Override maximum disparity when greater than upper bound
              - int
              - should be > min
              - None
              - No
            * - min_epi_tile_size
              - Lower bound of optimal epipolar tile size for dense matching
              - int
              - should be > 0
              - 300
              - No
            * - max_epi_tile_size
              - Upper bound of optimal epipolar tile size for dense matching
              - int
              - should be > 0 and > min
              - 1500
              - No
            * - epipolar_tile_margin_in_percent
              - Size of the margin used for dense matching (percent of tile size)
              - int
              -
              - 60
              - No
            * - performance_map_method
              - Compute performance map with selected method(s).
              - str, list, None
              - "risk", "intervals"
              - "risk"
              - No
            * - perf_eta_max_ambiguity
              - Ambiguity confidence eta max used for performance map (risk method)
              - float
              -
              - 0.99
              - No
            * - perf_eta_max_risk
              - Risk confidence eta max used for performance map (risk method)
              - float
              -
              - 0.25
              - No
            * - perf_eta_step
              - Risk and Ambiguity confidence eta step used for performance map (risk method)
              - float
              -
              - 0.04
              - No
            * - perf_ambiguity_threshold
              - Maximal ambiguity considered for performance map (risk method)
              - float
              -
              - 0.6
              - No
            * - save_intermediate_data
              - Save disparity map and disparity confidence
              - boolean
              -
              - false
              - No
            * - use_global_disp_range
              - If true, use global disparity range, otherwise local range estimation
              - boolean
              -
              - false
              - No
            * - local_disp_grid_step
              - Step of disparity min/ max grid used to resample dense disparity range
              - int
              -
              - 30
              - No
            * - disp_range_propagation_filter_size
              - Filter size of local min/max disparity, to propagate local min/max
              - int
              - should be > 0
              - 50
              -
            * - epi_disp_grid_tile_size
              - Tile size used for Disparity range grid generation.
              - int
              - should be > 0
              - 800
              - No
            * - use_cross_validation
              - Add cross validation step
              - bool, str
              - true, false, "fast", "accurate"
              - true
              - No
            * - denoise_disparity_map
              - Add disparity denoiser filter
              - bool
              -
              - false
              - No
            * - required_bands
              - Bands given to pandora
              - list
              - should be in input sensor
              - ["b0"]
              - No
            * - used_band
              - Band used for correlation
              - str
              - should be in input sensor
              - "b0"
              - No
            * - classification_fusion_margin
              - Margin for the fusion 
              - int 
              - should be > 0
              - -1
              - No
            * - threshold_disp_range_to_borders
              - Clip the disparity range to the valid region of right image
              - bool
              - 
              - False
              - No
            * - confidence_filtering
              - Parameters for the confidence filtering
              - dict
              - see below
              - see below
              - No

                
        See `Pandora documentation <https://pandora.readthedocs.io/>`_ for more information.

        Confidence filtering:

        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
        | Name                                 | Description                                                                                    | Type        | Available value        | Default value         | Required |
        +======================================+================================================================================================+=============+========================+=======================+==========+
        | activated                            | Activates filter of dense matches using confidence                                             | bool        |                        | True                  | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
        | bounds_ratio_threshold               | First filter : threshold for (bound_sup - bound_inf) / (disp_max - disp_min)                   | float       |                        | 0.2                   | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
        | risk_ratio_threshold                 | First filter : threshold for (risk_max - risk_min) / (disp_max - disp_min)                     | int         |                        | 0.8                   | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
        | win_nan_ratio                        | Second filter : window size for nan filtering                                                  | int         |                        | 20                    | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
        | nan_threshold                        | Second filter : threshold for the nan ratio (percentage of nan in the window)                  | float       |                        | 0.2                   | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
        | bounds_range_threshold               | Both filters : threshold for (bound_sup - bound_inf)                                           | int         |                        | 4                     | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
        | risk_range_threshold                 | Both filters : threshold for (risk_max - risk_min)                                             | int         |                        | 12                    | No       |
        +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+

        **Example**

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_dense_matching

        .. note::

            * Disparity range can be global (same disparity range used for each tile), or local (disparity range is estimated for each tile with dem min/max).
            * When user activate the generation of performance map, this map transits until being rasterized. Performance map is managed as a confidence map.
            * To save the confidence, the save_intermediate_data parameter should be activated.
            * The cross-validation step supports two modes: fast and accurate. Setting the configuration to true or "fast" will use the fast method, while setting it to "accurate" will enable the accurate method.
            * When setting the method to auto, cars will use a global classification map to select the optimal pandora configuration for dense matching

        .. list-table::
            :widths: 19 19
            :header-rows: 1

            * - Conf_name
              - Purpose
            * - census_sgm_default
              - This configuration is the one that works in most of cases using census 5 with sgm (p1 = 8, p2 = 32)
            * - mccnn_sgm
              - This configuration is the one that works in most of cases using mccnn with sgm (p1 = 2.3, p2 = 55.9)
            * - census_sgm_urban
              - This configuration is suitable for urban scene. It uses census11 with sgm (p1 = 20, p2 = 80)
            * - census_sgm_shadow
              - This configuration is suitable for shadow scene. It uses census11 with sgm (p1 = 20, p2 = 160)
            * - census_sgm_mountain_and_vegetation
              - This configuration is suitable for mountain or vegetation scene. It uses census11 with sgm (p1 = 38, p2 = 464)
            * - census_sgm_homogeneous
              - This configuration is suitable for homogeneous scene. It uses census11 with sgm (p1 = 72, p2 = 309)



    .. tab:: Dense match filling

        **Name**: "dense_match_filling"

        **Description**

        Fill holes in dense matches map.
        The zero_padding method fills the disparity with zeros where the selected classification values are non-zero values.

        **Configuration**

        +-------------------------------------+---------------------------------+-----------+-------------------------+--------------------+----------+
        | Name                                | Description                     | Type      | Available value         | Default value      | Required |
        +=====================================+=================================+===========+=========================+====================+==========+
        | method                              | Method for hole detection       | string    | "zero_padding"          | "zero_padding"     | No       |
        +-------------------------------------+---------------------------------+-----------+-------------------------+--------------------+----------+
        | save_intermediate_data              | Save disparity map              | boolean   |                         | False              | No       |
        +-------------------------------------+---------------------------------+-----------+-------------------------+--------------------+----------+
        | classification                      | Classification band name        | List[str] |                         | None               | No       |
        +-------------------------------------+---------------------------------+-----------+-------------------------+--------------------+----------+

        .. note::
            - The classification of second input is not given. Only the first disparity will be filled with zero value.
            - The filled area will be considered as a valid disparity mask.

        **Example**

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_dense_match_filling

    .. tab:: Triangulation

        **Name**: "triangulation"

        **Description**

        Triangulating the sights and get for each point of the reference image a latitude, longitude, altitude point

        **Configuration**

        +------------------------+--------------------------------------------------------------------------------------------------------------------+---------+--------------------------------------+------------------------------+----------+
        | Name                   | Description                                                                                                        | Type    | Available values                      | Default value               | Required |
        +========================+====================================================================================================================+=========+======================================+==============================+==========+
        | method                 | Method for triangulation                                                                                           | string  | "line_of_sight_intersection"         | "line_of_sight_intersection" | No       |
        +------------------------+--------------------------------------------------------------------------------------------------------------------+---------+--------------------------------------+------------------------------+----------+
        | snap_to_img1           | If all pairs share the same left image, modify lines of sight of secondary images to cross those of the ref image  | boolean |                                      | false                        | No       |
        +------------------------+--------------------------------------------------------------------------------------------------------------------+---------+--------------------------------------+------------------------------+----------+
        | save_intermediate_data | Save depth map as TIF, LAZ and CSV                                                                                 | boolean |                                      | false                        | No       |
        +------------------------+--------------------------------------------------------------------------------------------------------------------+---------+--------------------------------------+------------------------------+----------+

        **Example**

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_triangulation

    .. tab:: Point Cloud fusion

        **Name**: "point_cloud_fusion"

        **Description**

        Merge points clouds coming from each pair

        Only one method is available for now: "mapping_to_terrain_tiles"

        **Configuration**

        +------------------------------+------------------------------------------+---------+----------------------------+----------------------------+----------+
        | Name                         | Description                              | Type    | Available value            | Default value              | Required |
        +==============================+==========================================+=========+============================+============================+==========+
        | method                       | Method for fusion                        | string  | "mapping_to_terrain_tiles" | "mapping_to_terrain_tiles" | No       |
        +------------------------------+------------------------------------------+---------+----------------------------+----------------------------+----------+
        | save_intermediate_data       | Save points clouds as laz and csv format | boolean |                            | false                      | No       |
        +------------------------------+------------------------------------------+---------+----------------------------+----------------------------+----------+
        | save_by_pair                 | Enable points cloud saving by pair       | boolean |                            | false                      | No       |
        +------------------------------+------------------------------------------+---------+----------------------------+----------------------------+----------+

        **Example**


        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_point_cloud_fusion

        .. note::
            When `save_intermediate_data` is activated, multiple Laz and csv files are saved, corresponding to each processed terrain tiles.
            Please, see the section :ref:`merge_laz_files` to merge them into one single file.
            `save_by_pair` parameter enables saving by input pair. The csv/laz name aggregates row, col and corresponding pair key.

    .. tab:: Point Cloud outlier removal

        **Name**: "point_cloud_outlier_removal"

        **Description**

        Point cloud outlier removal

        **Configuration**

        +------------------------------+------------------------------------------+---------+-----------------------------------+---------------+----------+
        | Name                         | Description                              | Type    | Available value                   | Default value | Required |
        +==============================+==========================================+=========+===================================+===============+==========+
        | method                       | Method for point cloud outlier removal   | string  | "statistical", "small_components" | "statistical" | No       |
        +------------------------------+------------------------------------------+---------+-----------------------------------+---------------+----------+
        | save_intermediate_data       | Save points clouds as laz and csv format | boolean |                                   | false         | No       |
        +------------------------------+------------------------------------------+---------+-----------------------------------+---------------+----------+

        If method is *statistical*:

        +--------------------+-------------+---------+-----------------+---------------+----------+
        | Name               | Description | Type    | Available value | Default value | Required |
        +====================+=============+=========+=================+===============+==========+
        | k                  |             | int     | should be > 0   | 50            | No       |
        +--------------------+-------------+---------+-----------------+---------------+----------+
        | filtering_constant |             | float   | should be >= 0  | 0             | No       |
        +--------------------+-------------+---------+-----------------+---------------+----------+
        | mean_factor        |             | float   | should be >= 0  | 1.3           | No       |
        +--------------------+-------------+---------+-----------------+---------------+----------+
        | std_dev_factor     |             | float   | should be >= 0  | 3.0           | No       |
        +--------------------+-------------+---------+-----------------+---------------+----------+
        | use_median         |             | bool    |                 | True          | No       |
        +--------------------+-------------+---------+-----------------+---------------+----------+
        | half_epipolar_size |             | int     |                 | 5             | No       |
        +--------------------+-------------+---------+-----------------+---------------+----------+

        If method is *small_components*

        +---------------------------------+-------------+---------+-----------------+-----------------+----------+
        | Name                            | Description | Type    | Available value | Default value   | Required |
        +=================================+=============+=========+=================+=================+==========+
        | on_ground_margin                |             | int     |                 | 10              | No       |
        +---------------------------------+-------------+---------+-----------------+-----------------+----------+
        | connection_distance [#scaled]_  |             | float   |                 | None [#scaled]_ | No       |
        +---------------------------------+-------------+---------+-----------------+-----------------+----------+
        | nb_points_threshold             |             | int     |                 | 50              | No       |
        +---------------------------------+-------------+---------+-----------------+-----------------+----------+
        | clusters_distance_threshold     |             | float   |                 | None            | No       |
        +---------------------------------+-------------+---------+-----------------+-----------------+----------+
        | half_epipolar_size              |             | int     |                 | 5               | No       |
        +---------------------------------+-------------+---------+-----------------+-----------------+----------+

        .. warning::

            There is a particular case with the *Point Cloud outlier removal* application because both methods can be used at the same time in the pipeline.
            The ninth step consists of Filter the 3D points cloud via N consecutive filters.
            So you can configure the application any number of times. By default the filtering is done twice : once with the *small_components*, once with the *statistical* filter.
            To use your own filters in the order you want, you can add an identifier at the end of each application key :

            * *point_cloud_outlier_removal.my_first_filter*
            * *point_cloud_outlier_removal.filter_2*
            * *point_cloud_outlier_removal.3*

            The filtering steps will then be executed in the order you provided them.

            Because by default the applications *point_cloud_outlier_removal.1* and *point_cloud_outlier_removal.2* are defined, to not do any filtering you must set the configuration of *point_cloud_outlier_removal* to None.

        **Examples**

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_point_cloud_outlier_removal_1
        
        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_point_cloud_outlier_removal_2

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_point_cloud_outlier_removal_3

    .. tab:: Point Cloud Rasterization

        **Name**: "point_cloud_rasterization"

        **Description**

        Project altitudes on regular grid.

        Only one simple gaussian method is available for now.

        .. list-table:: Configuration
            :widths: 19 19 19 19 19 19
            :header-rows: 1

            * - Name
              - Description
              - Type
              - Available value
              - Default value
              - Required
            * - method
              -
              - string
              - "simple_gaussian"
              - simple_gaussian
              - No
            * - dsm_radius
              -
              - float, int
              -
              - 1.0
              - No
            * - sigma
              -
              - float
              -
              - None
              - No
            * - grid_points_division_factor
              -
              - int
              -
              - None
              - No
            * - dsm_no_data
              -
              - int
              -
              - -32768
              -
            * - texture_no_data
              - If texture_no_data is None, it will be automatically set to the maximum value of texture_dtype
              - int, None
              -
              - None
              -
            * - texture_dtype
              - By default, it's retrieved from the input texture. Otherwise, specify an image type
              - string
              - "uint8", "uint16", "float32" ...
              - None
              - No
            * - msk_no_data
              - No data value for mask  and classif
              - int
              -
              - 255
              -
            * - save_intermediate_data
              - Save all layers from input point cloud in application `dump_dir`
              - boolean
              -
              - false
              - No

        **Example**

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_point_cloud_rasterization


    .. tab:: DSM Filling

        **Name**: "dsm_filling"

        **Description**

        Fill classified values or missing values with one the three avalable methods.

        **Configuration**

        +-------------------------------------+---------------------------------+---------+----------------------------------------------------------+--------------------+----------+
        | Name                                | Description                     | Type    | Available value                                          | Default value      | Required |
        +=====================================+=================================+=========+==========================================================+====================+==========+
        | method                              | Method for hole detection       | string  | "exogenous_filling", "bulldozer", "border_interpolation" |                    | Yes      |
        +-------------------------------------+---------------------------------+---------+----------------------------------------------------------+--------------------+----------+
        | save_intermediate_data              | Save disparity map              | boolean |                                                          | False              | No       |
        +-------------------------------------+---------------------------------+---------+----------------------------------------------------------+--------------------+----------+


        **Method exogenous_filling:**

        Method "exogenous_filling" fills with altitude of exogenous data (DEM/geoid).

        +-------------------------------------+----------------------------------------------------+-------------+-------------------------+--------------------+----------+
        | Name                                | Description                                        | Type        | Available value         | Default value      | Required |
        +=====================================+====================================================+=============+=========================+====================+==========+
        | classification                      | Classification band name                           | List[str]   |                         | "nodata"           | No       |
        +-------------------------------------+----------------------------------------------------+-------------+-------------------------+--------------------+----------+
        | fill_with_geoid                     | Classes to fill with geoid                         | List[str]   |                         | None               | No       |
        +-------------------------------------+----------------------------------------------------+-------------+-------------------------+--------------------+----------+
        | interpolation_method                | Interpolation method for DEM and geoid resampling  | List[str]   | "bilinear", "cubic"     | None               | No       |
        +-------------------------------------+----------------------------------------------------+-------------+-------------------------+--------------------+----------+


        **Method bulldozer:**

        Method "bulldozer" converts the DSM to a DTM and fills the pixels with the output DTM.

        +-------------------------------------+---------------------------------+-----------+-------------------------+--------------------+----------+
        | Name                                | Description                     | Type      | Available value         | Default value      | Required |
        +=====================================+=================================+===========+=========================+====================+==========+
        | classification                      | Classification band name        | List[str] |                         | "nodata"           | No       |
        +-------------------------------------+---------------------------------+-----------+-------------------------+--------------------+----------+

        **Method border_interpolation:**

        Method "border_interpolation" use the border of every component to compute the altitude to fill.

        +-------------------------------------+------------------------------------------+-----------+-------------------------+--------------------+----------+
        | Name                                | Description                              | Type      | Available value         | Default value      | Required |
        +=====================================+==========================================+===========+=========================+====================+==========+
        | classification                      | Classification band name                 | List[str] |                         | "nodata"           | No       |
        +-------------------------------------+------------------------------------------+-----------+-------------------------+--------------------+----------+
        | component_min_size                  | Minimal size (pixels) of feature to fill | int       |                         | 5                  | No       |
        +-------------------------------------+------------------------------------------+-----------+-------------------------+--------------------+----------+
        | border_size                         | Size of border used to estimate altitude | int       |                         | 10                 | No       |
        +-------------------------------------+------------------------------------------+-----------+-------------------------+--------------------+----------+
        | percentile                          | Percentile of border taken for altitude  | float     |                         | 10                 | No       |
        +-------------------------------------+------------------------------------------+-----------+-------------------------+--------------------+----------+

        .. note::
            - If the keyword "nodata" is added to the classification band name parameter, nodata pixels of the classification will be filled. If no classification is given, nodata pixels of DSM will be filled.

        .. warning::

            There is a particular case with the *dsm_filling* application because it can be called any number of times.
            Because it is not possible to define three times the *dsm_filling* in your yaml/json configuration file, you can add an identifier after *dsm_filling* to differentiate each application :

            * *dsm_filling.border_interp*
            * *dsm_filling.two*
            * *dsm_filling.with_bulldozer*

            It is recommended to run bulldozer before border_interpolation in order for border_interpolation to get a DTM. If no DTM is found, border_interpolation will use the DSM.
            The execution order is determined by the order of the applications in the configuration file.

        **Example**

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_dsm_filling_1

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_dsm_filling_2

        .. include-cars-config:: ../../example_configs/how_to_use_CARS/advanced_configuration/applications_dsm_filling_3

    .. tab:: Auxiliary Filling

        **Name**: "auxiliary_filling"

        **Description**

        Fill in the missing values of the texture and classification by using information from sensor inputs 
        This application replaces the existing `texture.tif` and `classification.tif`.
        
        The application retrieves texture and classification information by performing inverse location on the input sensor images. It is therefore necessary to provide the `sensors` category in `inputs` configuration in order to use this application, even when `depth_map` are provided as input. The pairing information is also required: when searching for texture information, the application will always look in the first sensor of the pair and then in the second, if no information for the given pixel is found in the first sensor. The final filled value of the pixel is the average of the contribution of each pair. The classification information is a logical OR of all classifications.

        In `fill_nan` mode, only the pixels that are no-data in the auxiliary images that are valid in the reference dsm will be filled while in full mode all valid pixel from the reference dsm are filled.

        If `use_mask` is set to `true`, the texture data from a sensor will not be used if the corresponding sensor mask value is false. If the pixel is masked in all images, the filled texture will be the average of the first sensor texture of each pair

        When ``save_intermediate_data`` is activated, the folder ``dump_dir/auxiliary_filling`` will contain the non-filled texture and classification.

        **Configuration**

        +------------------------------+---------------------------------------------+---------+----------------------------------+----------------------------------+----------+
        | Name                         | Description                                 | Type    | Available values                 | Default value                    | Required |
        +==============================+=============================================+=========+==================================+==================================+==========+
        | method                       | Method for filling                          | string  | "auxiliary_filling_from_sensors" | "auxiliary_filling_from_sensors" | No       |
        +------------------------------+---------------------------------------------+---------+----------------------------------+----------------------------------+----------+
        | activated                    | Activates the filling                       | boolean |                                  | false                            | No       |
        +------------------------------+---------------------------------------------+---------+----------------------------------+----------------------------------+----------+
        | mode                         | Processing mode                             | string  | "fill_nan", "full"               | false                            | No       |
        +------------------------------+---------------------------------------------+---------+----------------------------------+----------------------------------+----------+
        | use_mask                     | Use mask information from input sensors     | boolean |                                  | true                             | No       |
        +------------------------------+---------------------------------------------+---------+----------------------------------+----------------------------------+----------+
        | texture_interpolator         | Interpolator used for texture interpolation | string  | "linear", "nearest", "cubic"     | "linear"                         | No       |
        +------------------------------+---------------------------------------------+---------+----------------------------------+----------------------------------+----------+
        | save_intermediate_data       | Saves the temporary data in dump_dir        | boolean |                                  | false                            | No       |
        +------------------------------+---------------------------------------------+---------+----------------------------------+----------------------------------+----------+

.. rubric:: Footnotes

.. [#scaled] This parameter is computed at runtime depending on the resolution of the input sensor images. You can still override it in the configuration.