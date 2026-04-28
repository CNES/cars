.. _sparse_matching_app:

Sparse matching
===============

**Name**: "sparse_matching"

**Description**

Compute keypoints matches on pair images

**Architecture Note**

This application uses a plugin-based architecture separating the application layer from the method layer:

- **Application Parameters**: These control the parallelization at the application level (tiling, margins, match validation, etc.) and are independent of the matching algorithm used.
- **Method Parameters**: These are algorithm-specific parameters (e.g., SIFT tuning parameters) and belong to the selected method plugin.

The `application` parameter selects which parallelization strategy to use.
The `method` parameter selects which matching algorithm/preset to use.

Both application and method may have their own parameters, which should then be put all together in the `sparse_matching` configuration key.

Applications
~~~~~~~~~~~~

Basic Application
-----------------

+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
| Name                                 | Description                                                                                    | Type        | Available value           | Default value | Required |
+======================================+================================================================================================+=============+===========================+===============+==========+
| application                          | Application to use in the pipeline                                                             | string      | "basic"                   | "basic"       | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
| method                               | Method to use for sparse matching                                                              | string      |  "sift"                   | "sift"        | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
| elevation_delta_lower_bound          | Expected lower bound for elevation delta with respect to input low resolution dem in meters    | int, float  |                           | None          | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
| elevation_delta_upper_bound          | Expected upper bound for elevation delta with respect to input low resolution dem in meters    | int, float  |                           | None          | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
| epipolar_error_upper_bound           | Expected upper bound for epipolar error in pixels                                              | float, str  | should be > 0             | auto          | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
| epipolar_error_estimation            | Mean for epipolar error in pixels                                                              | float, str  | should be >= 0            | auto          | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
| epipolar_error_maximum_bias          | Maximum bias for epipolar error in pixels                                                      | float, str  | should be >= 0            | auto          | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
| save_intermediate_data               | Save matches in epipolar geometry (4 first columns) and sensor geometry (4 last columns)       | boolean     |                           | false         | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
| tile_margin                          | Margin to use on tiles                                                                         | int         | should be > 0             | 10            | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
| minimum_nb_matches                   | Minimum number of matches that must be computed to continue pipeline                           | int         | should be > 0             | 90            | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
| decimation_factor                    | Reduce the number of keypoints                                                                 | int         | should be > 0             | 30            | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
| disparity_bounds_estimation          | Parameters for the estimation of disparity interval                                            | dict        |                           | {}            | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+

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

.. note::

    'elevation_delta_lower_bound' and 'elevation_delta_upper_bound' are overidden to  [-1000, 9000] in default pipeline if no initial elevation is set.
    If initial elevation is set, it is overridden to [-500, 1000].

Methods
~~~~~~~

SIFT Method
-----------

**Name**: "sift"

**Description**: Scale-Invariant Feature Transform (SIFT) based sparse matching

The SIFT method parameters are automatically selected from the sparse matching configuration above when method is set to "sift".

Method-specific Parameters:

+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
| Name                                 | Description                                                                                    | Type        | Available value           | Default value | Required |
+======================================+================================================================================================+=============+===========================+===============+==========+
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
| sift_back_matching                   | Also check that right vs. left gives same match                                                | boolean     |                           | true          | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
| used_band                            | Name of band used for correlation                                                              | int         | should be in input sensor | "b0"          | No       |
+--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+

For more information about SIFT parameters, please refer to the `VLFEAT SIFT documentation <https://www.vlfeat.org/api/sift.html>`_.

.. note::
    For the decimation factor, a value of 33 means that we divide the number of sift by 3, a value of 100 means that we do not decimate them


Examples
~~~~~~~~

Minimal example:

.. include-cars-config:: ../../example_configs/configuration/applications_sparse_matching_minimal

Example with both application and method parameters:

.. include-cars-config:: ../../example_configs/configuration/applications_sparse_matching_full
