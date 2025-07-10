.. _advanced configuration:

Advanced configuration
======================

This section describes CARS main advanced configuration structure through a `json <http://www.json.org/json-fr.html>`_ configuration file.

The structure follows this organization:

.. code-block:: json

    {
        "inputs": {},
        "orchestrator": {},
        "applications": {},
        "output": {},
        "advanced": {}
    }

.. warning::

    Be careful with commas to separate each section. None needed for the last json element.

.. tabs::

    .. tab:: Applications

        This key is optional and is used to redefine parameters for each application used in the pipeline. 
        You can personnalize the configuration for each resolution at which the pipeline is ran, or override the parameters for all resolutions at once, as explained in the section right below. 

        .. tabs::

            .. tab:: Overriding all resolutions at once

                This is the default behaviour when providing a configuration dict directly in the `applications` key.

                This example overrides the configuration of `application_name` for all resolutions at once :

                .. code-block:: json

                    "applications": {
                        "application_name": {
                            "method": "application_dependent",
                            "parameter1": 3,
                            "parameter2": 0.3
                        }
                    }


            .. tab:: Overriding a single resolution

                To override a configuration at a specific resolution, you first need to identify which resolution you want to modify. By default, CARS uses the resolutions 16, 4, and 1.

                Once you have chosen the resolution value, you can override the configuration by adding an entry to the `applications` dictionary with the key `resolution_{resolution_value}`.

                The following example overrides the configuration for `application_name` at resolutions 4 and 1, using different parameters for each. Resolution 16 will retain its default configuration.

                .. code-block:: json

                    "applications": {
                        "resolution_4": {
                            "application_name": {
                                "method": "first_method",
                                "parameter1": 26,
                                "parameter2": 0.9
                            }
                        },
                        "resolution_1": {
                            "application_name": {
                                "method": "second_method",
                                "parameter1": 8,
                                "parameter2": 0.2
                            }
                        }
                    }

        By default, the configuration can be different for the first resolution, the intermediate resolution(s) and the last resolution. 
        The changes to the default values can be modified in the source code, in ``cars/pipelines/conf_resolution/*``.

        The section below includes the files directly.

        .. tabs::

            .. tab:: Overriding configuration : first resolution

                This is empty for now.
              
                .. include:: ../../../cars/pipelines/conf_resolution/conf_first_resolution.json
                    :literal:

            .. tab:: Overriding configuration : all intermediate resolutions
              
                This is empty for now.

                .. include:: ../../../cars/pipelines/conf_resolution/conf_intermediate_resolution.json
                    :literal:

            .. tab:: Overriding configuration : final resolution
              
                .. include:: ../../../cars/pipelines/conf_resolution/conf_final_resolution.json
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

                .. code-block:: json

                    "applications": {
                        "grid_generation": {
                            "method": "epipolar",
                            "epi_step": 35
                        }
                    },

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

                .. code-block:: json

                    "applications": {
                        "resampling": {
                            "method": "bicubic",
                            "epi_tile_size": 600
                        }
                    },

            .. tab:: Sparse matching

                **Name**: "sparse_matching"

                **Description**

                Compute keypoints matches on pair images

                **Common parameters**

                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
                | Name                                 | Description                                                                                    | Type        | Available value           | Default value | Required |
                +======================================+================================================================================================+=============+===========================+===============+==========+
                | disparity_margin                     | Add a margin to min and max disparity as percent of the disparity range.                       | float       |                           | 0.02          | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
                | epipolar_error_upper_bound           | Expected upper bound for epipolar error in pixels                                              | float       | should be > 0             | 10.0          | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+---------------------------+---------------+----------+
                | epipolar_error_maximum_bias          | Maximum bias for epipolar error in pixels                                                      | float       | should be >= 0            | 0.0           | No       |
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


                .. note::

                    'elevation_delta_lower_bound' and 'elevation_delta_upper_bound' are overidden to  [-1000, 9000] in default pipeline if no initial elevation is set.
                    If initial elevation is set, it is overridden to [-500, 1000].

                **Sift:**

                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-------------------------------------------+----------+
                | Name                                 | Description                                                                                    | Type        | Available value        | Default value                             | Required |
                +======================================+================================================================================================+=============+========================+===========================================+==========+
                | sift_matching_threshold              | Threshold for the ratio to nearest second match                                                | float       | should be > 0          | 0.7                                       | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-------------------------------------------+----------+
                | sift_n_octave                        | The number of octaves of the Difference of Gaussians scale space                               | int         | should be > 0          | 8                                         | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-------------------------------------------+----------+
                | sift_n_scale_per_octave              | The numbers of levels per octave of the Difference of Gaussians scale space                    | int         | should be > 0          | 3                                         | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-------------------------------------------+----------+
                | sift_peak_threshold                  | Constrast threshold to discard a match (at None it will be set according to image type)        | float       | should be > 0          | 4.0                                       | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-------------------------------------------+----------+
                | sift_edge_threshold                  | Distance to image edge threshold to discard a match                                            | float       |                        | 10.0                                      | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-------------------------------------------+----------+
                | sift_magnification                   | The descriptor magnification factor                                                            | float       | should be > 0          | 7.0                                       | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-------------------------------------------+----------+
                | sift_window_size                     | smaller values let the center of the descriptor count more                                     | int         | should be > 0          | 2                                         | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-------------------------------------------+----------+
                | decimation_factor                    | Reduce the number of sifts                                                                     | int         | should be > 0          | 30                                        | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-------------------------------------------+----------+

                For more information about these parameters, please refer to the `VLFEAT SIFT documentation <https://www.vlfeat.org/api/sift.html>`_.

                .. note::
                    For the decimation factor, a value of 33 means that we divide the number of sift by 3, a value of 100 means that we do not decimate them


                **Pandora:**

                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | Name                                 | Description                                                                                    | Type        | Available value        | Default value         | Required |
                +======================================+================================================================================================+=============+========================+=======================+==========+
                | resolution                           | Resolution at which the image will be downsampled for the use of pandora                       | int, list   | should be > 0          | 4                     | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | loader_conf                          | Pandora configuration that will be used                                                        | dict        |                        | Pandora default conf  | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | connection_val                       | distance to use to consider that two points are connected                                      | float       | should be > 0          | 3.0                   | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | nb_pts_threshold                     | number of points to use to identify small clusters to filter                                   | int         | should be > 0          | 80                    | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | filtered_elt_pos                     | if filtered_elt_pos is set to True, the removed points positions in their original \           |             |                        |                       |          |
                |                                      | epipolar images are returned, otherwise it is set to None                                      | bool        |                        | False                 | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | clusters_distance_threshold          | distance to use to consider if two points clusters are far from each other or not              | float       |                        | None                  | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | confidence_filtering                 | parameters for the confidence filtering                                                        | dict        |                        | True                  | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | "disparity_bounds_estimation"        | parameters for the estimation of disparity interval                                            | dict        |                        | True                  | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+

                Confidence filtering:

                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | Name                                 | Description                                                                                    | Type        | Available value        | Default value         | Required |
                +======================================+================================================================================================+=============+========================+=======================+==========+
                | activated                            | activates filter of matches using confidence                                                   | bool        |                        | True                  | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | upper_bound                          | the upper bound for the intervals bound sup confidence                                         | int         |                        | 5                     | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | lower_bound                          | the lower bound for the intervals bound sup confidence                                         | int         |                        | -20                   | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | risk_max                             | the maximum risk that is accepted in the mean risk_max confidence                              | int         |                        | 60                    | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | nan_threshold                        | the threshold for the nanratio confidence (percentage of nan around a pixel)                   | float       |                        | 0.1                   | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | win_nanratio                         | the window size for the nanratio computation using generic_filter                              | int         |                        | 20                    | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | win_mean_risk_max                    | the window size for the mean risk max computation using generic_filter                         | int         |                        | 7                     | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+

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


                .. warning::

                    Because it is not possible to define twice the *application_name* on your json configuration file, we have decided to configure
                    those two applications with :

                    * *sparse_matching.sift*
                    * *sparse_matching.pandora*

                    Each one is associated to a particular *sparse_matching* method.
                    Therefore, is it not possible to use the key *sparse_matching* and to select the method.


                **Example**

                .. code-block:: json

                    "applications": {
                        "sparse_matching.sift": {
                            "method": "sift",
                            "disparity_margin": 0.01
                        },
                        "sparse_matching.pandora":{
                            "method": "pandora",
                            "resolution": [4, 2]
                        }
                    },

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
                | height_margin                   | Height margin [margin min, margin max], in meter                         | int        |                                      | 20                     | No       |
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

                +---------------------------------+--------------------------------------------------------------------------+------------+-----------------+---------------+----------+
                | Name                            | Description                                                              | Type       | Available value | Default value | Required |
                +=================================+==========================================================================+============+=================+===============+==========+
                | resolution                      | Resolution of dem, in meter                                              | int, float | should be > 0   | 90            | No       |
                +---------------------------------+--------------------------------------------------------------------------+------------+-----------------+---------------+----------+
                | margin                          | Margin to use on the border of dem, in meter                             | int, float | should be > 0   | 6000          | No       |
                +---------------------------------+--------------------------------------------------------------------------+------------+-----------------+---------------+----------+
                | fillnodata_max_search_distance  | Max search distance for rasterio fill nodata                             | int        | should be > 0   | 3             | No       |
                +---------------------------------+--------------------------------------------------------------------------+------------+-----------------+---------------+----------+
                | percentile                      | Percentile of matches to ignore in min and max functions                 | int        | should be > 0   | 1             | No       |
                +---------------------------------+--------------------------------------------------------------------------+------------+-----------------+---------------+----------+
                | min_number_matches              | Minimum number of matches needed to have a valid tile                    | int        | should be > 0   | 30            | No       |
                +---------------------------------+--------------------------------------------------------------------------+------------+-----------------+---------------+----------+

                **Method bulldozer_on_raster**

                Rasterizes all matches on a regular grid and performs morphological operations and Bulldozer processing to compute DEM min and max

                +---------------------------------+--------------------------------------------------------------------------+------------+-----------------+---------------+----------+
                | Name                            | Description                                                              | Type       | Available value | Default value | Required |
                +=================================+==========================================================================+============+=================+===============+==========+
                | resolution                      | Resolution of dem, in meter                                              | int, float | should be > 0   | 90            | No       |
                +---------------------------------+--------------------------------------------------------------------------+------------+-----------------+---------------+----------+
                | margin                          | Margin to use on the border of dem, in meter                             | int, float | should be > 0   | 500           | No       |
                +---------------------------------+--------------------------------------------------------------------------+------------+-----------------+---------------+----------+
                | fillnodata_max_search_distance  | Max search distance for rasterio fill nodata                             | int        | should be > 0   | 50            | No       |
                +---------------------------------+--------------------------------------------------------------------------+------------+-----------------+---------------+----------+
                | morphological_filters_size      | Size (in pixels) of erosion and dilation filters used to generate DEM    | int        | should be > 0   | 30            | No       |
                +---------------------------------+--------------------------------------------------------------------------+------------+-----------------+---------------+----------+
                | median_filter_size              | Size (in pixels) of median filter used to generate median DEM            | int        | should be > 0   | 5             | No       |
                +---------------------------------+--------------------------------------------------------------------------+------------+-----------------+---------------+----------+
                | dem_median_output_resolution    | Resolution of output downsampled median DEM                              | int        | should be > 0   | 30            | No       |
                +---------------------------------+--------------------------------------------------------------------------+------------+-----------------+---------------+----------+
                | bulldozer_max_object_size       | Bulldozer parameter "max_object_size"                                    | int        | should be > 0   | 16            | No       |
                +---------------------------------+--------------------------------------------------------------------------+------------+-----------------+---------------+----------+
                | compute_stats                   | Compute statistics of difference between DEM min/max and original DSM    | boolean    |                 | true          | No       |
                +---------------------------------+--------------------------------------------------------------------------+------------+-----------------+---------------+----------+

                **Example**

                .. code-block:: json

                    "applications": {
                        "dem_generation": {
                            "method": "dichotomic",
                            "min_number_matches": 20
                        }
                    }

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

                .. code-block:: json

                    "applications": {
                        "ground_truth_reprojection": {
                            "method": "direct_loc",
                            "target": "all"
                        }
                    }

                .. figure:: ../images/cars_pipeline_advanced.png
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
                      - "census_sgm_default", "mccnn_sgm", "census_sgm_urban", "census_sgm_shadow", "census_sgm_mountain_and_vegetation", "census_sgm_homogeneous"
                      - "census_sgm_default"
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
                      - 300
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

                See `Pandora documentation <https://pandora.readthedocs.io/>`_ for more information.

                **Example**

                .. code-block:: json

                    "applications": {
                        "dense_matching": {
                            "method": "census_sgm_default",
                            "loader": "pandora",
                            "loader_conf": "path_to_user_pandora_configuration"
                        }
                    },

                .. note::

                    * Disparity range can be global (same disparity range used for each tile), or local (disparity range is estimated for each tile with dem min/max).
                    * When user activate the generation of performance map, this map transits until being rasterized. Performance map is managed as a confidence map.
                    * To save the confidence, the save_intermediate_data parameter should be activated.
                    * The cross-validation step supports two modes: fast and accurate. Setting the configuration to true or "fast" will use the fast method, while setting it to "accurate" will enable the accurate method.

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

                Fill holes in dense matches map. This uses the holes detected with the HoleDetection application.
                The holes correspond to the area masked for dense matching.

                **Configuration**

                +-------------------------------------+---------------------------------+---------+-------------------------+--------------------+----------+
                | Name                                | Description                     | Type    | Available value         | Default value      | Required |
                +=====================================+=================================+=========+=========================+====================+==========+
                | method                              | Method for hole detection       | string  | "plane", "zero_padding" | "plane"            | No       |
                +-------------------------------------+---------------------------------+---------+-------------------------+--------------------+----------+
                | save_intermediate_data              | Save disparity map              | boolean |                         | False              | No       |
                +-------------------------------------+---------------------------------+---------+-------------------------+--------------------+----------+


                **Method plane:**

                +-------------------------------------+---------------------------------+-------------+-------------------------+--------------------+----------+
                | Name                                | Description                     | Type        | Available value         | Default value      | Required |
                +=====================================+=================================+=============+=========================+====================+==========+
                | classification                      | Classification band name        | List[str]   |                         | None               | No       |
                +-------------------------------------+---------------------------------+-------------+-------------------------+--------------------+----------+
                | interpolation_type                  | Interpolation type              | string      | "pandora"               | "pandora"          | No       |
                +-------------------------------------+---------------------------------+-------------+-------------------------+--------------------+----------+
                | interpolation_method                | Method for hole interpolation   | string      | "mc_cnn"                | "mc_cnn"           | No       |
                +-------------------------------------+---------------------------------+-------------+-------------------------+--------------------+----------+
                | max_search_distance                 | Maximum search distance         | int         |                         | 100                | No       |
                +-------------------------------------+---------------------------------+-------------+-------------------------+--------------------+----------+
                | smoothing_iterations                | Number of smoothing iterations  | int         |                         | 1                  | No       |
                +-------------------------------------+---------------------------------+-------------+-------------------------+--------------------+----------+
                | ignore_nodata_at_disp_mask_borders  | Ignore nodata at borders        | boolean     |                         | false              | No       |
                +-------------------------------------+---------------------------------+-------------+-------------------------+--------------------+----------+
                | ignore_zero_fill_disp_mask_values   | Ignore zeros                    | boolean     |                         | true               | No       |
                +-------------------------------------+---------------------------------+-------------+-------------------------+--------------------+----------+
                | ignore_extrema_disp_values          | Ignore extrema values           | boolean     |                         | true               | No       |
                +-------------------------------------+---------------------------------+-------------+-------------------------+--------------------+----------+
                | nb_pix                              | Margin used for mask            | int         |                         | 20                 | No       |
                +-------------------------------------+---------------------------------+-------------+-------------------------+--------------------+----------+
                | percent_to_erode                    | Percentage to erode             | float       |                         | 0.2                | No       |
                +-------------------------------------+---------------------------------+-------------+-------------------------+--------------------+----------+


                **Method zero_padding:**

                The zero_padding method fills the disparity with zeros where the selected classification values are non-zero values.

                +-------------------------------------+---------------------------------+-----------+-------------------------+--------------------+----------+
                | Name                                | Description                     | Type      | Available value         | Default value      | Required |
                +=====================================+=================================+===========+=========================+====================+==========+
                | classification                      | Classification band name        | List[str] |                         | None               | No       |
                +-------------------------------------+---------------------------------+-----------+-------------------------+--------------------+----------+

                .. note::
                    - The classification of second input is not given. Only the first disparity will be filled with zero value.
                    - The filled area will be considered as a valid disparity mask.

                .. warning::

                    There is a particular case with the *dense_match_filling* application because it is called twice.
                    The eighth step consists of fill dense matches via two consecutive methods.
                    So you can configure the application twice , once for the *plane*, the other for *zero_padding* method.
                    Because it is not possible to define twice the *application_name* on your json configuration file, we have decided to configure
                    those two applications with :

                    * *dense_match_filling.1*
                    * *dense_match_filling.2*

                    Each one is associated to a particular *dense_match_filling* method*
                    Therefore, is it not possible to use the key *dense_match_filling* and to select the method.

                **Example**

                .. code-block:: json

                        "applications": {
                            "dense_match_filling.1": {
                                "method": "plane",
                                "classification": ["water"],
                                "save_intermediate_data": true
                            },
                            "dense_match_filling.2": {
                                "method": "zero_padding",
                                "classification": ["cloud", "snow"],
                                "save_intermediate_data": true
                            }
                        },


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

                .. code-block:: json

                    "applications": {
                        "triangulation": {
                            "method": "line_of_sight_intersection",
                            "snap_to_img1": true
                        }
                    },

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


                .. code-block:: json

                        "applications": {
                            "point_cloud_fusion": {
                                "method": "mapping_to_terrain_tiles",
                                "save_intermediate_data": true,
                                "save_by_pair": true,
                            }
                        },

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
                | activated          |             | boolean |                 | True          | No       |
                +--------------------+-------------+---------+-----------------+---------------+----------+
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

                +-----------------------------+-------------+---------+-----------------+---------------+----------+
                | Name                        | Description | Type    | Available value | Default value | Required |
                +=============================+=============+=========+=================+===============+==========+
                | activated                   |             | boolean |                 | True          | No       |
                +-----------------------------+-------------+---------+-----------------+---------------+----------+
                | on_ground_margin            |             | int     |                 | 10            | No       |
                +-----------------------------+-------------+---------+-----------------+---------------+----------+
                | connection_distance         |             | float   |                 | 3.0           | No       |
                +-----------------------------+-------------+---------+-----------------+---------------+----------+
                | nb_points_threshold         |             | int     |                 | 50            | No       |
                +-----------------------------+-------------+---------+-----------------+---------------+----------+
                | clusters_distance_threshold |             | float   |                 | None          | No       |
                +-----------------------------+-------------+---------+-----------------+---------------+----------+
                | half_epipolar_size          |             | int     |                 | 5             | No       |
                +-----------------------------+-------------+---------+-----------------+---------------+----------+

                .. warning::

                    There is a particular case with the *Point Cloud outlier removal* application because it is called twice.
                    The ninth step consists of Filter the 3D points cloud via two consecutive filters.
                    So you can configure the application twice , once for the *small component filters*, the other for *statistical* filter.
                    Because it is not possible to define twice the *application_name* on your json configuration file, we have decided to configure
                    those two applications with :

                    * *point_cloud_outlier_removal.1*
                    * *point_cloud_outlier_removal.2*

                    Each one is associated to a particular *point_cloud_outlier_removal* method*
                    Therefore, is it not possible to use the key *point_cloud_outlier_removal* and to select the method.


                **Example**

                .. code-block:: json

                    "applications": {
                        "point_cloud_outlier_removal.1": {
                            "method": "small_components",
                            "on_ground_margin": 10,
                            "save_intermediate_data": true,
                        },
                        "point_cloud_outlier_removal.2": {
                            "method": "statistical",
                            "k": 10,
                            "save_intermediate_data": true,
                        }
                    }

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
                      -
                      - int
                      -
                      - 0
                      -
                    * - texture_dtype
                      - | By default, it's retrieved from the input texture
                        | Otherwise, specify an image type
                      - string
                      - | "uint8", "uint16"
                        | "float32" ...
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

                .. code-block:: json

                    "applications": {
                        "point_cloud_rasterization": {
                            "method": "simple_gaussian",
                            "dsm_radius": 1.5
                        }
                    },


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
                | activated                           | Activate this application                          | bool        |                         | False              | No       |
                +-------------------------------------+----------------------------------------------------+-------------+-------------------------+--------------------+----------+
                | classification                      | Classification band name                           | List[str]   |                         | None               | No       |
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
                | activated                           | Activate this application       | bool      |                         | False              | No       |
                +-------------------------------------+---------------------------------+-----------+-------------------------+--------------------+----------+
                | classification                      | Classification band name        | List[str] |                         | None               | No       |
                +-------------------------------------+---------------------------------+-----------+-------------------------+--------------------+----------+

                **Method border_interpolation:**

                Method "border_interpolation" use the border of every component to compute the altitude to fill.

                +-------------------------------------+------------------------------------------+-----------+-------------------------+--------------------+----------+
                | Name                                | Description                              | Type      | Available value         | Default value      | Required |
                +=====================================+==========================================+===========+=========================+====================+==========+
                | activated                           | Activate this application                | bool      |                         | False              | No       |
                +-------------------------------------+------------------------------------------+-----------+-------------------------+--------------------+----------+
                | classification                      | Classification band name                 | List[str] |                         | None               | No       |
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

                    There is a particular case with the *dsm_filling* application because it is called three times.
                    Because it is not possible to define three times the *dsm_filling* on your json configuration file, we have decided to configure
                    those three applications with :

                    * *dsm_filling.1*
                    * *dsm_filling.2*
                    * *dsm_filling.3*

                    Each one is associated to a particular *dsm_filling* method : 
                     - 1 : exogenous_filling
                     - 2 : bulldozer
                     - 3 : border_interpolation

                    It is not recommended to change it, as the pipeline is designed with this order. If you just want to use a subset of these applications, just use the "activate" parameter.
                    It is recommended to run bulldozer before border_interpolation in order for border_interpolation to get a DTM. If no DTM is found, border_interpolation will use the DSM.

                **Example**

                .. code-block:: json

                  "applications": {
                    "dsm_filling.1": {
                        "method": "exogenous_filling",
                        "activated": true,
                        "classification": ["sea"],
                        "fill_with_geoid": ["sea"],
                        "save_intermediate_data": true
                    },
                    "dsm_filling.2": {
                        "method": "bulldozer",
                        "activated": true,
                        "classification": ["cloud"],
                        "save_intermediate_data": true
                    },
                    "dsm_filling.3": {
                        "method": "border_interpolation",
                        "activated": true,
                        "classification": ["lake"],
                        "save_intermediate_data": true
                    }
                  }

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



    .. tab:: Advanced parameters

        Here are the advanced parameters. This key is optional and can be useful if you want to use CARS more as a developer.

        .. list-table:: Configuration
            :widths: 19 19 19 19 19
            :header-rows: 1

            * - Name
              - Description
              - Type
              - Default value
              - Required
            * - save_intermediate_data
              - Save intermediate data for all applications, at any or all resolutions
              - bool or dict[bool]
              - False
              - No
            * - keep_low_res_dir
              - Whether to save the output of all resolution runs or not 
              - bool
              - true
              - No
            * - use_epipolar_a_priori
              - Active epipolar a priori
              - bool
              - False
              - Yes
            * - epipolar_a_priori
              - Provide epipolar a priori information (see section below)
              - dict
              -
              - No
            * - terrain_a_priori
              - Provide terrain a priori information (see section below)
              - dict
              -
              - No
            * - epipolar_resolutions
              - The resolutions at which the Unit Pipeline will be ran for each pair
              - list[int]
              - [16, 4, 1]
              - No
            * - debug_with_roi
              - Use input ROI with the tiling of the entire image (see Inputs section)
              - bool
              - False
              - No
            * - merging
              - Merge point clouds before rasterization (soon to be deprecated)
              - bool
              - False
              - No
            * - dsm_merging_tile_size
              - Tile size to use in dsms merging
              - int
              - 4000
              - No
            * - performance_map_classes
              - List defining interval: [a,b,c,d] generates [[a,b],[b,c],[c,d]] intervals used in the performance map classification. If null, raw performance map is given
              - list or None
              - [0, 0.968, 1.13375, 1.295, 1.604, 2.423, 3.428]
              - No
            * - ground_truth_dsm
              - Datas to be reprojected from the application ground_truth_reprojection
              - dict
              -
              - No
            * - phasing
              - Phase to use for DSM {"point" : (x,y) , "epsg": epsg}
              - dict
              -
              - No
            * - geometry_plugin
              - Name of the geometry plugin to use and optional parameters
              - str or dict
              - "SharelocGeometry"
              - No
            * - pipeline
              - Name of the pipeline to use
              - str
              - "default"
              - No
            * - texture_bands
              - Name of the bands used for output ortho image (see Sensor loaders configuration for details)
              - list
              - None
              - No


        .. tabs::
	
            .. tab:: Save intermediate data

                The `save_intermediate_data` flag can be used to activate and deactivate the saving of the possible output of applications.

                It is set in the `advanced` category and can be overloaded in each application separately. It defaults to false, meaning that no intermediate product in saved. 
                Intermediate data are saved in the `dump_dir` folder found in CARS output directory, with a subfolder corresponding to each application.

                For example, setting `save_intermediate_data` to `true` in `advanced` and to `false` in `applications/point_cloud_rasterization` will activate product saving in all applications except `point_cloud_rasterization`.
                Conversely, setting it to `false` in `advanced` and to `true` in `applications/point_cloud_rasterization` will only save rasterization outputs.

                Intermediate data refers to all files that are not part of an output product. Files that compose an output product will not be found in the application dump directory.
                For exemple if `dsm` is requested as output product, the `dsm.tif` files and all activated dsm auxiliary files will not be found in `rasterization` dump directory.
                This directory will still contain the files generated by the `rasterization` application that are not part of the `dsm` product.

                `save_intermediate_data` can be either a dict or a bool.
                A bool will enable `save_intermediate_data` for all resolutions.
                A dict will enable it for any resolution where it's marked as true, and disable it for any resolution where it's marked as false (or isn't in the dict).

                The following example enables `save_intermediate_data` for all applications at all resolutions : 

                .. code-block:: json

                    "advanced": {
                        "save_intermediate_data": true
                    }

                This in turn will produce the following folder structure :

                .. code-block::

                    cars_output_folder/
                        dsm/
                        dump_dir/
                        intermediate_res/
                            out_res16/
                                dsm/
                                dump_dir/
                            out_res4/
                                dsm/
                                dump_dir/
                
                The following example enables `save_intermediate_data` for all applications at resolution 16, while keeping it disabled for both resolution 4 and resolution 1 :

                .. code-block:: json

                    "advanced": {
                        "save_intermediate_data": {
                            "resolution_16": true,
                            "resolution_1": false
                        }
                    }


            .. tab:: Keep low res dir

                The `keep_low_res_dir` parameter flag can be used to specify that you would like the intermediate DSMs and DEMs to be saved in their respective directory.

                By default, since `keep_low_res_dir` is true, you will find the intermediate DSMs and DEMs in `intermediate_res/out_res{resolution_value}/dsm`.
                If `save_intermediate_data` was enabled for an application of an intermediate resolution, those results will be found in `intermediate_res/out_res{resolution_value}/dump_dir`.

                The following example disables the saving of all intermediate resolutions' outputs :

                .. code-block:: json

                    "advanced": {
                        "keep_low_res_dir": false
                    }
              
            .. tab:: Epipolar a priori

                The CARS pipeline produces a ``used_conf.json`` in the `outdir` that contains the `epipolar_a_priori`
                information for each sensor image pairs. If you wish to re-run CARS, this time by skipping the
                sparse matching, you can use the ``used_conf.json`` as the new input configuration, with
                its `use_epipolar_a_priori` parameter set to `True`.

                For each sensor images, the epipolar a priori are filled as following:

                +-----------------------+-------------------------------------------------------------+--------+----------------+----------------------------------+
                | Name                  | Description                                                 | Type   | Default value  | Required                         |
                +=======================+=============================================================+========+================+==================================+
                | *grid_correction*     | The grid correction coefficients                            | list   |                | if use_epipolar_a_priori is True |
                +-----------------------+-------------------------------------------------------------+--------+----------------+----------------------------------+
                | *disparity_range*     | The disparity range [disp_min, disp_max]                    | list   |                | if use_epipolar_a_priori is True |
                +-----------------------+-------------------------------------------------------------+--------+----------------+----------------------------------+

                .. note::

                    The grid correction coefficients are based on bilinear model with 6 parameters [x1,x2,x3,y1,y2,y3].
                    The None value produces no grid correction (equivalent to parameters [0,0,0,0,0,0]).


            .. tab:: Terrain a priori

                The `terrain_a_priori` is used at the same time that `epipolar_a_priori`.
                If `use_epipolar_a_priori` is activated, `epipolar_a_priori` and `terrain_a_priori` must be provided.
                The terrain_a_priori data dict is produced during low or full resolution dsm pipeline.

                The terrain a priori is initially populated with DEM information.

                +----------------+-------------------------------------------------------------+--------+----------------+----------------------------------+
                | Name           | Description                                                 | Type   | Default value  | Required                         |
                +================+=============================================================+========+================+==================================+
                | *dem_median*   | DEM generated with median function                          | str    |                | if use_epipolar_a_priori is True |
                +----------------+-------------------------------------------------------------+--------+----------------+----------------------------------+
                | *dem_min*      | DEM generated with min function                             | str    |                | if use_epipolar_a_priori is True |
                +----------------+-------------------------------------------------------------+--------+----------------+----------------------------------+
                | *dem_max*      | DEM generated with max function                             | str    |                | if use_epipolar_a_priori is True |
                +----------------+-------------------------------------------------------------+--------+----------------+----------------------------------+

            .. tab:: Epipolar resolutions

                The `epipolar_resolutions` parameter is used to specify the number and resolution of Unit Pipeline runs.
                Resolutions are set from the lowest to the highest, with 1 being the heighest possible.
                A resolution of n means that one pixel from the downsampled image will be calculated using n² pixels from the full-res image.
                
                For example, epipolar_resolutions = [16, 4, 2, 1] with an image of 2048x3072 will run the Unit Pipeline four times :

                - First with a size of 128x192
                - Then with a resolution of 512x768
                - Then with a resolution of 1024x1536
                - And a last time with a resolution of 2048x3072

                Each run will provide an apriori on the height of the terrain at each position for the next run, resulting in a low computation time.


            .. tab:: Ground truth DSM

                To activate the ground truth reprojection application, it is necessary to specify the required inputs in the advanced settings.
                For this, a dictionary named `ground_truth_dsm` must be added, containing the keys presented in the following table.
                By default, the used dsm is considered on ellipsoid. If not, fill the `geoid` parameter.

				+---------------------------------+------------------------------------------------------------+--------------------+------------------------------+-------------------------------------------------------+----------+
				| Name                            | Description                                                | Type               | Available value              | Default value                                         | Required |
				+=================================+============================================================+====================+==============================+=======================================================+==========+
				| dsm                             | Path to ground truth dsm (Lidar for example)               | string             |                              |                                                       | Yes      |
				+---------------------------------+------------------------------------------------------------+--------------------+------------------------------+-------------------------------------------------------+----------+
				| geoid                           | DSM geoid.                                                 | bool or string     |                              |  False                                                | No       |
				+---------------------------------+------------------------------------------------------------+--------------------+------------------------------+-------------------------------------------------------+----------+
				| auxiliary_data                  | The lidar auxiliaries data                                 | dict               |                              |  None                                                 | No       |
				+---------------------------------+------------------------------------------------------------+--------------------+------------------------------+-------------------------------------------------------+----------+
				| auxiliary_data_interpolation    | The lidar auxiliaries data interpolator                    | dict               |                              |  None (nearest if auxiliary_data is not None)         | No       |
				+---------------------------------+------------------------------------------------------------+--------------------+------------------------------+-------------------------------------------------------+----------+

				.. note::

					The parameter `geoid` refers to the vertical reference of the ground truth DSM. It can be set as a string to provide the path to a geoid file on disk, or as a boolean: if set to True CARS default geoid is used, if set to False no vertical offset is applied (ellipsoid reference).

                Example:

                .. code-block:: json

                    "advanced":
                        {
                            "ground_truth_dsm": {
                                "dsm": "path/to/ground/truth/dsm.tif",
								"auxiliary_data":{
									"classification": "path/to/classification.tif",
									"texture": "path/to/texture.tif"
								},
								"auxiliary_data_interpolation":{
									"classification": "nearest",
									"texture": "linear"
								}
                            }
                        }

            .. tab:: Phasing

                Phase can be added to make sure multiple DSMs can be merged in "dsm -> dsm" pipeline.
                "point" and "epsg" of point must be specified

                +-------------------+--------------------------+----------------+-------------------------+---------------------------------------+----------+
                | Name              | Description              | Type           | Default value           | Available values                      | Required |
                +===================+==========================+================+=========================+=======================================+==========+
                | *point*           | Point to phase on        | tuple          | None                    |                                       | False    |
                +-------------------+--------------------------+----------------+-------------------------+---------------------------------------+----------+
                | *epsg*            | Epsg of point            | int            | None                    |                                       | False    |
                +-------------------+--------------------------+----------------+-------------------------+---------------------------------------+----------+

                 .. code-block:: json

                          "phasing": {
                              "point": [32000, 30000],
                              "epsg": 32530
                          }

            .. tab:: Geometry plugin

                This section describes configuration of the geometry plugins for CARS, please refer to :ref:`plugins` section for details on plugins installation.

                +-------------------+-----------------------+----------------+-------------------------+---------------------------------------+----------+
                | Name              | Description           | Type           | Default value           | Available values                      | Required |
                +===================+=======================+================+=========================+=======================================+==========+
                | *geometry_plugin* | The plugin to use     | str or dict    | "SharelocGeometry"      | "SharelocGeometry"                    | False    |
                +-------------------+-----------------------+----------------+-------------------------+---------------------------------------+----------+

                **geometry_plugin** allow user to specify other parameters, through a dictionary:

                +-------------------+--------------------------+----------------+-------------------------+---------------------------------------+----------+
                | Name              | Description              | Type           | Default value           | Available values                      | Required |
                +===================+==========================+================+=========================+=======================================+==========+
                | *plugin_name*     | The plugin name to use   | str            | "SharelocGeometry"      | "SharelocGeometry"                    | False    |
                +-------------------+--------------------------+----------------+-------------------------+---------------------------------------+----------+
                | *interpolator*    | Interpolator to use      | str            | "cubic"                 | "cubic" , "linear"                    | False    |
                +-------------------+--------------------------+----------------+-------------------------+---------------------------------------+----------+


                To use Shareloc geometry library, CARS input configuration should be defined as :

                .. code-block:: json

                    {
                        "inputs": {
                        "sensors": {
                          "one": {
                            "image": "img1.tif",
                            "geomodel": {
                              "path": "img1.geom",
                              "model_type": "RPC"
                            },
                          },
                          "two": {
                            "image": "img2.tif",
                            "geomodel": {
                              "path": "img2.geom",
                              "model_type": "RPC"
                            },
                          }
                        },
                        "pairing": [["one", "two"]],
                        "initial_elevation": {
                            "dem": "path/to/srtm_file.tif"
                          },
                        },
                        "advanced":{
                            "geometry_plugin": "SharelocGeometry"
                        }
                    }

                **geometry_plugin** specify the plugin to use, but other configuration parameters can be specified :

                .. code-block:: json

                        "advanced":{
                            "geometry_plugin": {
                                "plugin_name": "SharelocGeometry",
                                "interpolator": "cubic"
                            }
                        }

                The particularities in the configuration file are:

                * **geomodel.model_type**: Depending on the nature of the geometric models indicated above, this field as to be defined as :term:`RPC` or `GRID`. By default, "RPC".
                * **initial_elevation**: Field contains the path to the **file** corresponding the srtm tiles covering the production (and **not** a directory !!)
                * **geometry_plugin**: Parameter configured to "SharelocGeometry" to use Shareloc plugin.

                Parameter can also be defined as a string *path* instead of a dictionary in the configuration. In this case, geomodel parameter will
                be changed to a dictionary before launching the pipeline. The dictionary will be :

                .. code-block:: json

                    {
                      "path": "img1.geom",
                      "model_type": "RPC"
                    }

                .. note::

                    Be aware that geometric models must therefore be opened by Shareloc directly in this case, and supported sensors may evolve.

            .. tab:: Pipeline configurations
                The ``pipeline`` key is optional and allows users to choose the pipeline they would like to run. By default, CARS has a single pipeline: `default`.
                This pipeline is modular and can be adapted to your needs. This sections provides examples of specific configurations.

                Installed plugins may provide additional pipelines. The inputs and outputs are specific to each pipeline. This section describes the pipeline available in CARS.

                +----------------+-----------------------+--------+---------------+------------------+----------+
                | Name           | Description           | Type   | Default value | Available values | Required |
                +================+=======================+========+===============+==================+==========+
                | *pipeline*     | The pipeline to use   | str    | "default"     | "default"        | False    |
                +----------------+-----------------------+--------+---------------+------------------+----------+

                .. code-block:: json

                      "advanced": {
                          "pipeline": "your_pipeline_name"
                          }
                      }

    .. tab:: Sensor loaders

        Sensor loaders are used to read images and classifications on sensor geometry with an advanced level on configuration. They are used inside the Inputs configuration (see :ref:`basic configuration`).

        Two sensor loaders are available in CARS : "basic" and "pivot".

        .. tabs::

            .. tab:: Basic loader 

                The basic loader is the simplest way to define an image. The basic loader is the one used by default when only a path is given. However, it is possible to use the basic loader with a dictionary : 

                +----------------+-----------------------+--------+---------------+------------------+----------+
                | Name           | Description           | Type   | Default value | Available values | Required |
                +================+=======================+========+===============+==================+==========+
                | *loader*       | Name of sensor loader | str    | "basic"       | "basic"          | False    |
                +----------------+-----------------------+--------+---------------+------------------+----------+
                | *path*         | File path             | str    |               |                  | True     |
                +----------------+-----------------------+--------+---------------+------------------+----------+
                | *no_data*      | No data value of file | int    | 0             |                  | False    |
                +----------------+-----------------------+--------+---------------+------------------+----------+
      
            .. tab:: Pivot loader 

                The pivot loader allows the maximal level of configuration. To use the pivot loader, it is required to set the "loader" parameter in sensor loader configuration.

                +-----------------+---------------------------------------------------------------------------------------+--------+-------------------+------------------+----------+
                | Name            | Description                                                                           | Type   | Default value     | Available values | Required |
                +=================+=======================================================================================+========+===================+==================+==========+
                | *loader*        | Name of sensor loader                                                                 | str    | "pivot"           | "pivot"          | True     |
                +-----------------+---------------------------------------------------------------------------------------+--------+-------------------+------------------+----------+
                | *main_file*     | Main file path among the files given in `bands` parameter                             | str    | File of band "b0" |                  | False    |
                +-----------------+---------------------------------------------------------------------------------------+--------+-------------------+------------------+----------+
                | *bands*         | Dictionary listing for every band of the image, the corresponding file and band index | int    |                   |                  | True     |
                +-----------------+---------------------------------------------------------------------------------------+--------+-------------------+------------------+----------+
                | *texture_bands* | List of bands used for output ortho image                                             | list   | None              |                  | False    |
                +-----------------+---------------------------------------------------------------------------------------+--------+-------------------+------------------+----------+
                | *no_data*       | No data value of file                                                                 | int    | 0                 |                  | False    |
                +-----------------+---------------------------------------------------------------------------------------+--------+-------------------+------------------+----------+

                The `bands` dictionary have keys which correspond to name of bands. The name of bands is imposed by CARS : if the image has n bands, the name of the bands must be ["b0", "b1", ..., "b{n-1}"].
                Each key points to a dictionary with keys "path" and "band_id".

                With the pivot format, an image can be composed of several files.

                Full configuration example for pivot sensor loader :

                .. code-block:: json

                    "image": {
                      "loader": "pivot",
                      "main_file": "img1.tif",
                      "bands": {
                        "b0": {
                          "path": "img1.tif",
                          "band": 0
                        },
                        "b1": {
                          "path": "color1.tif",
                          "band": 0
                        },
                        "b2": {
                          "path": "color1.tif",
                          "band": 1
                        },
                        "b3": {
                          "path": "color1.tif",
                          "band": 2
                        }
                      },
                      "texture_bands": ["b1", "b2", "b3"]
                    }

                .. note::

                     - In the above example, the texture bands correspond to the three bands of `color1.tif` which is a RGB file, so the output `texture.tif` will be RGB.
                     - Order matters : if the "texture_bands" parameter is set to ["b3", "b2", "b1"], the output will be BGR.
                     - It is possible to fuse the different files in output ortho image : if the "texture_bands" parameter is set to ["b0", "b3", "b2", "b1"], the output will be PBGR (with P from panchromatic).
                     - If "texture_bands" parameter is None (default value), all bands will be texture bands, so the output will be PRGB.
                     - Parameter "texture_bands" must be the same as the one defined in Advanced parameters. If multiple pairs are used in the configuration, every left image must have the same texture bands in order to fuse them.

                Documentation on plugin creation can be found in :ref:`creating_a_plugin`

                


