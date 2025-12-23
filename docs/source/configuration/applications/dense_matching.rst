.. _dense_matching_app:

Dense matching
==============

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
      - 10
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
    * - filter_incomplete_disparity_range
      - Removes pixels whose disparity range is not fully valid pixels
      - bool
      -
      - True
      - No
    * - generate_ambiguity
      - Generate the ambiguity
      - bool
      -
      - False
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
| risk_ratio_threshold                 | First filter : threshold for (risk_max - risk_min) / (disp_max - disp_min)                     | int         |                        | 0.75                  | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
| win_nan_ratio                        | Second filter : window size for nan filtering                                                  | int         |                        | 20                    | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
| nan_threshold                        | Second filter : threshold for the nan ratio (percentage of nan in the window)                  | float       |                        | 0.2                   | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
| bounds_range_threshold               | Both filters : threshold for (bound_sup - bound_inf)                                           | int         |                        | 3                     | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
| risk_range_threshold                 | Both filters : threshold for (risk_max - risk_min)                                             | int         |                        | 9                    | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+

**Example**

.. include-cars-config:: ../../example_configs/configuration/applications_dense_matching

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
