.. _dense_matching_app:

Dense matching
==============

**Name**: "dense_matching"

**Description**

Compute the disparity map from stereo-rectified pair images

**Architecture Note**

This application uses a plugin-based architecture separating the application layer from the method layer:

- **Application Parameters**: These control the orchestration and tiling of the dense matching process (tile sizes, disparity range estimation, global settings, etc.) and are independent of the matching algorithm used.
- **Method Parameters**: These are algorithm-specific parameters (e.g., Pandora confidence filtering, cross-validation mode, etc.) and belong to the selected method plugin.

The `application` parameter selects which parallelization strategy to use.
The `method` parameter selects which matching algorithm/preset to use.

Both application and method may have their own parameters, which should then be put all together in the `dense_matching` configuration key.

Applications
~~~~~~~~~~~~

Basic Application
-----------------

.. list-table:: Configuration
    :widths: 19 19 19 19 19 19
    :header-rows: 1

    * - Name
      - Description
      - Type
      - Available value
      - Default value
      - Required
    * - application
      - Application to use
      - string
      - "basic"
      - "basic"
      - No
    * - method
      - Method for dense matching
      - string
      - see Methods section below
      - "pandora_auto"
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
    * - required_bands
      - Bands required by the dense matching application
      - list
      - should be in input sensor
      - ["b0"]
      - No

.. note::

    * ``use_global_disp_range``: Disparity range can be global (same disparity range used for each tile), or local (disparity range is estimated for each tile with dem min/max).
    * To save the confidence, the save_intermediate_data parameter should be activated.

Methods
~~~~~~~

Pandora Method
--------------

**Names**: "pandora_custom", "pandora_mccnn_sgm", "pandora_census_sgm_urban", "pandora_census_sgm_shadow", "pandora_census_sgm_mountain_and_vegetation", "pandora_census_sgm_homogeneous", "pandora_census_sgm_default", "pandora_census_sgm_sparse", "pandora_auto"

**Description**: Dense matching method using Pandora, with various presets available for different scenes

**Available Method Presets**:

.. list-table::
    :widths: 25 75
    :header-rows: 1

    * - Method Name
      - Description
    * - pandora_auto
      - Automatic method selection based on global classification map
    * - pandora_custom
      - Uses the custom Pandora configuration defined in loader_conf
    * - pandora_census_sgm_default
      - Default configuration using Census 5 with SGM (p1 = 8, p2 = 32), works in most cases
    * - pandora_mccnn_sgm
      - MCCNN with SGM (p1 = 2.3, p2 = 55.9)
    * - pandora_census_sgm_urban
      - Optimized for urban scenes using Census 11 with SGM (p1 = 20, p2 = 80)
    * - pandora_census_sgm_shadow
      - Optimized for scenes with shadows using Census 11 with SGM (p1 = 20, p2 = 160)
    * - pandora_census_sgm_mountain_and_vegetation
      - Optimized for mountainous or vegetation scenes using Census 11 with SGM (p1 = 38, p2 = 464)
    * - pandora_census_sgm_homogeneous
      - Optimized for homogeneous scenes using Census 11 with SGM (p1 = 72, p2 = 309)

**Method-specific Parameters**:

+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| Name                                 | Description                                                                                    | Type        | Available value                 | Default value         | Required |
+======================================+================================================================================================+=============+=================================+=======================+==========+
| generate_ambiguity                   | Generate the ambiguity map                                                                     | bool        |                                 | False                 | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| performance_map_method               | Compute performance map with selected method(s)                                                | str, list   | "risk", "intervals"             | None                  | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| perf_eta_max_ambiguity               | Ambiguity confidence eta max used for performance map (risk method)                            | float       |                                 | 0.99                  | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| perf_eta_max_risk                    | Risk confidence eta max used for performance map (risk method)                                 | float       |                                 | 0.25                  | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| perf_eta_step                        | Risk and Ambiguity confidence eta step used for performance map (risk method)                  | float       |                                 | 0.04                  | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| perf_ambiguity_threshold             | Maximal ambiguity considered for performance map (risk method)                                 | float       |                                 | 0.6                   | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| classification_fusion_margin         | Margin for the fusion                                                                          | int         |                                 | -1                    | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| use_cross_validation                 | Add cross validation step                                                                      | bool, str   | true, false, "fast", "accurate" | "fast"                | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| denoise_disparity_map                | Add disparity denoiser filter                                                                  | bool        |                                 | false                 | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| used_band                            | Band used for correlation                                                                      | str         | should be in input sensor       | "b0"                  | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| loader                               | External library used to compute dense matching                                                | str         | "pandora"                       | "pandora"             | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| loader_conf                          | Configuration for Pandora loader (for pandora_custom method)                                   | dict or str |                                 | None                  | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| confidence_filtering                 | Parameters for dense match filtering using confidence                                          | dict        | see details table below         | {}                    | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| threshold_disp_range_to_borders      | Clip the disparity range to the valid region of right image                                    | bool        |                                 | False                 | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| filter_incomplete_disparity_range    | Removes pixels whose disparity range is not fully valid                                        | bool        |                                 | True                  | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| edges_3sgm                           | Use 3SGM in Pandora with edge mask as mode (when edges_mask is given as input)                 | bool        |                                 | True                  | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+
| classification_3sgm                  | Use 3SGM in Pandora with classification as mode (when classification is given as input)        | list[int]   | list of integer class IDs       | None                  | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------------+-----------------------+----------+

.. note::

    * When user activate the generation of performance map, this map transits until being rasterized. Performance map is managed as a confidence map.
    * The cross-validation step supports two modes: fast and accurate. Setting the configuration to true or "fast" will use the fast method, while setting it to "accurate" will enable the accurate method.
    * When setting the method to pandora_auto, CARS will use a global classification map to select the optimal pandora configuration for dense matching.
    * The ``classification_3sgm`` parameter is used over ``edges_3sgm`` when both are activated, as both can't be used simultaneously.

The following table details the method-specific ``confidence_filtering`` parameter of Pandora.

Pandora confidence_filtering parameter details:

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
| risk_range_threshold                 | Both filters : threshold for (risk_max - risk_min)                                             | int         |                        | 9                     | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+

Examples
~~~~~~~~

Minimal example:

.. include-cars-config:: ../../example_configs/configuration/applications_dense_matching_minimal

Example with both application and method parameters:

.. include-cars-config:: ../../example_configs/configuration/applications_dense_matching_full

Example using a custom Pandora preset:

.. include-cars-config:: ../../example_configs/configuration/applications_dense_matching_custom
