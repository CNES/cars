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

        This key is optional and allows to redefine parameters for each application used in pipeline.

        This section describes all possible configuration of CARS applications.

        CARS applications are defined and called by their name in applications configuration section:

        .. code-block:: json

            "applications":{
                "application_name": {
                    "method": "application_dependent",
                    "parameter1": 3,
                    "parameter2": 0.3
                }
            }


        Be careful with these parameters: no mechanism ensures consistency between applications for now.
        And some parameters can degrade performance and DSM quality heavily.
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
                | save_intermediate_data | Save epipolar images and color                         | boolean |                 | false         | No       |
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

                **Configuration**

                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | Name                                 | Description                                                                                    | Type        | Available value        | Default value | Required |
                +======================================+================================================================================================+=============+========================+===============+==========+
                | method                               | Method for sparse matching                                                                     | string      | "sift", "pandora"      | "sift"        | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | disparity_margin                     | Add a margin to min and max disparity as percent of the disparity range.                       | float       |                        | 0.02          | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | epipolar_error_upper_bound           | Expected upper bound for epipolar error in pixels                                              | float       | should be > 0          | 10.0          | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | epipolar_error_maximum_bias          | Maximum bias for epipolar error in pixels                                                      | float       | should be >= 0         | 0.0           | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | sift_back_matching                   | Also check that right vs. left gives same match                                                | boolean     |                        | true          | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | matches_filter_knn                   | Number of neighbors used to measure isolation of matches and detect isolated matches           | int         | should be > 0          | 25            | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | matches_filter_dev_factor            | Factor of deviation of isolation of matches to compute threshold of outliers                   | int, float  | should be > 0          | 3.0           | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | save_intermediate_data               | Save matches in epipolar geometry (4 first columns) and sensor geometry (4 last columns)       | boolean     |                        | false         | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | strip_margin                         | Margin to use on strip                                                                         | int         | should be > 0          | 10            | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | elevation_delta_lower_bound          | Expected lower bound for elevation delta with respect to input low resolution dem in meters    | int, float  |                        | -9000         | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | elevation_delta_upper_bound          | Expected upper bound for elevation delta with respect to input low resolution dem in meters    | int, float  |                        | 9000          | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | minimum_nb_matches                   | Minimum number of matches that must be computed to continue pipeline                           | int         | should be > 0          | 100           | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+


                **Sift:**

                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-------------------------------------------+----------+
                | Name                                 | Description                                                                                    | Type        | Available value        | Default value                             | Required |
                +======================================+================================================================================================+=============+========================+===========================================+==========+
                | disparity_outliers_rejection_percent | Percentage of outliers to reject                                                               | float       | between 0 and 1        | 0.1                                       | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-------------------------------------------+----------+
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
                | decimation_factor                    | Reduce the number of sifts                                                                     | int         | should be > 0          | 20 if pandora is activated, 100 otherwise | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-------------------------------------------+----------+

                For more information about these parameters, please refer to the `VLFEAT SIFT documentation <https://www.vlfeat.org/api/sift.html>`_.

                .. note::
                    For the decimation factor, a value of 33 means that we divide the number of sift by 3, a value of 100 means that we do not decimate them


                **Pandora:**

                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | Name                                 | Description                                                                                    | Type        | Available value        | Default value         | Required |
                +======================================+================================================================================================+=============+========================+=======================+==========+
                | resolution                           | Resolution at which the image will be downsampled for the use of pandora                       | int, list   |    should be > 0       | 4                     | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | loader_conf                          | Pandora configuration that will be used                                                        | dict        |                        | Pandora default conf  | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | connection_val                       | distance to use to consider that two points are connected                                      | float       | should be > 0          | 3.0                   | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | nb_pts_threshold                     |number of points to use to identify small clusters to filter                                    | int         | should be > 0          | 80                    | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | filtered_elt_pos                     | if filtered_elt_pos is set to True, the removed points positions in their original \           |             |                        |                       |          |
                |                                      | epipolar images are returned, otherwise it is set to None                                      | bool        |                        | False                 | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+
                | clusters_distance_threshold          | distance to use to consider if two points clusters are far from each other or not              | float       |                        | None                  | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+-----------------------+----------+

                .. warning::

                    There is a particular case with the *sparse_matching* application because it can be called twice.
                    So you can configure the application twice , once for the *sift*, the other for *pandora* method.
                    Because it is not possible to define twice the *application_name* on your json configuration file, we have decided to configure
                    those two applications with :

                    *sparse_matching.sift*
                    *sparse_matching.pandora*

                    Each one is associated to a particular *sparse_matching* method*
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

                .. note::
                    * Sift will always be used during the cars execution
                    * Pandora is optionnal, by default this one is not activated
                    * You can use both sift and pandora during your execution, the combined matches will be used

            .. tab:: DEM Generation

                **Name**: "dem_generation"

                **Description**

                Generates dem from sparse matches.

                3 dems are generated, with different methods:
                * median
                * min
                * max

                The DEMs are generated in the application dump directory

                **Configuration**

                +---------------------------------+------------------------------------------------------------+------------+-----------------+---------------+----------+
                | Name                            | Description                                                | Type       | Available value | Default value | Required |
                +=================================+============================================================+============+=================+===============+==========+
                | method                          | Method for dem_generation                                  | string     | "dichotomic"    | "dichotomic"  | Yes      |
                +---------------------------------+------------------------------------------------------------+------------+-----------------+---------------+----------+
                | resolution                      | Resolution of dem, in meter                                | int, float |  should be > 0  | 200           | No       |
                +---------------------------------+------------------------------------------------------------+------------+-----------------+---------------+----------+
                | margin                          | Margin to use on the border of dem, in meter               | int, float |  should be > 0  | 6000          | No       |
                +---------------------------------+------------------------------------------------------------+------------+-----------------+---------------+----------+
                | percentile                      | Percentile of matches to ignore in min and max functions   | int        | should be > 0   | 3             | No       |
                +---------------------------------+------------------------------------------------------------+------------+-----------------+---------------+----------+
                | min_number_matches              | Minimum number of matches needed to have a valid tile      | int        | should be > 0   | 30            | No       |
                +---------------------------------+------------------------------------------------------------+------------+-----------------+---------------+----------+
                | height_margin                   | Height margin [margin min, margin max], in meter           | int        |                 | 20            | No       |
                +---------------------------------+------------------------------------------------------------+------------+-----------------+---------------+----------+
                | fillnodata_max_search_distance  | Max search distance for rasterio fill nodata               | int        | should be > 0   | 3             | No       |
                +---------------------------------+------------------------------------------------------------+------------+-----------------+---------------+----------+
                | min_dem                         | Min value that has to be reached by dem_min                | int        | should be < 0   | -500          | No       |
                +---------------------------------+------------------------------------------------------------+------------+-----------------+---------------+----------+
                | max_dem                         | Max value that has to be reached by dem_max                | int        | should be > 0   | 1000          | No       |
                +---------------------------------+------------------------------------------------------------+------------+-----------------+---------------+----------+
                | save_intermediate_data          | Save DEM as TIF                                            | boolean    |                 | false         | No       |
                +---------------------------------+------------------------------------------------------------+------------+-----------------+---------------+----------+

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
                      - "census_sgm", "mccnn_sgm"
                      - "census_sgm"
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
                    * - generate_performance_map
                      - Generate a performance map from disparity map
                      - boolean
                      -
                      - False
                      - No
                    * - generate_confidence_intervals
                      - Compute confidence intervals from disparity map.
                      - boolean
                      -
                      - False
                      - No
                    * - perf_eta_max_ambiguity
                      - Ambiguity confidence eta max used for performance map
                      - float
                      -
                      - 0.99
                      - No
                    * - perf_eta_max_risk
                      - Risk confidence eta max used for performance map
                      - float
                      -
                      - 0.25
                      - No
                    * - perf_eta_step
                      - Risk and Ambiguity confidence eta step used for performance map
                      - float
                      -
                      - 0.04
                      - No
                    * - perf_ambiguity_threshold
                      - Maximal ambiguity considered for performance map
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
                      - bool
                      -
                      - false
                      - No

                See `Pandora documentation <https://pandora.readthedocs.io/>`_ for more information.

                **Example**

                .. code-block:: json

                    "applications": {
                        "dense_matching": {
                            "method": "census_sgm",
                            "loader": "pandora",
                            "loader_conf": "path_to_user_pandora_configuration"
                        }
                    },

                .. note::

                    * Disparity range can be global (same disparity range used for each tile), or local (disparity range is estimated for each tile with dem min/max).
                    * When user activate the generation of performance map, this map transits until being rasterized. Performance map is managed as a confidence map.
                    * To save the confidence, the save_intermediate_data parameter should be activated.


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
                | activated          |             | boolean |                 | false         | No       |
                +--------------------+-------------+---------+-----------------+---------------+----------+
                | k                  |             | int     | should be > 0   | 50            | No       |
                +--------------------+-------------+---------+-----------------+---------------+----------+
                | std_dev_factor     |             | float   | should be > 0   | 5.0           | No       |
                +--------------------+-------------+---------+-----------------+---------------+----------+
                | use_median         |             | bool    |                 | True          | No       |
                +--------------------+-------------+---------+-----------------+---------------+----------+
                | half_epipolar_size |             | int     |                 | 5             | No       |
                +--------------------+-------------+---------+-----------------+---------------+----------+

                If method is *small_components*

                +-----------------------------+-------------+---------+-----------------+---------------+----------+
                | Name                        | Description | Type    | Available value | Default value | Required |
                +=============================+=============+=========+=================+===============+==========+
                | activated                   |             | boolean |                 | false         | No       |
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
                            "activated": true
                        },
                        "point_cloud_outlier_removal.2": {
                            "method": "statistical",
                            "k": 10,
                            "save_intermediate_data": true,
                            "activated": true
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
                    * - color_no_data
                      -
                      - int
                      -
                      - 0
                      -
                    * - color_dtype
                      - | By default, it's retrieved from the input color
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

                Fill in the missing values of the DSM by using the DEM's elevation.
                This application replaces the existing dsm.tif.

                Only one method is available for now: "bulldozer".

                .. note::

                    When ``save_intermediate_data`` is activated, the folder ``dump_dir/dsm_filling`` will contain :

                    * The replaced dsm.tif, saved under ``dump_dir/dsm_filling/dsm_not_filled.tif``
                    * The dsm given to Bulldozer as input, saved under ``dump_dir/dsm_filling/dsm_filled_with_dem_not_smoothed.tif``
                    * The configuration given to Bulldozer, saved under ``dump_dir/dsm_filling/bulldozer_config.yaml``
                    * All the outputs generated by Bulldozer, saved under ``dump_dir/dsm_filling/bulldozer/``


                **Configuration**

                +------------------------------+-----------------------------------------+---------+----------------------------+----------------------------+----------+
                | Name                         | Description                             | Type    | Available values           | Default value              | Required |
                +==============================+=========================================+=========+============================+============================+==========+
                | method                       | Method for filling                      | string  | "bulldozer"                | "bulldozer"                | No       |
                +------------------------------+-----------------------------------------+---------+----------------------------+----------------------------+----------+
                | activated                    | Activates the filling                   | boolean |                            | false                      | No       |
                +------------------------------+-----------------------------------------+---------+----------------------------+----------------------------+----------+
                | save_intermediate_data       | Saves the temporary data in dump_dir    | boolean |                            | false                      | No       |
                +------------------------------+-----------------------------------------+---------+----------------------------+----------------------------+----------+

                **Example**


                .. code-block:: json

                        "applications": {
                            "dsm_filling": {
                                "method": "bulldozer",
                                "activated": true,
                            }
                        },

            .. tab:: Auxiliary Filling

                **Name**: "auxiliary_filling"

                **Description**

                Fill in the missing values of the color and classification by using information from sensor inputs 
                This application replaces the existing `color.tif` and `classification.tif`.
                
                The application retrieves color and classification information by performing inverse location on the input sensor images. It is therefore necessary to provide the `sensors` category in `inputs` configuration in order to use this application, even when `depth_map` are provided as input. The pairing information is also required: when searching for color information, the application will always look in the first sensor of the pair and then in the second, if no information for the given pixel is found in the first sensor. The final filled value of the pixel is the average of the contribution of each pair. The classification information is a logical OR of all classifications.

                In `fill_nan` mode, only the pixels that are no-data in the auxiliary images that are valid in the reference dsm will be filled while in full mode all valid pixel from the reference dsm are filled.

                If `use_mask` is set to `true`, the color data from a sensor will not be used if the corresponding sensor mask value is false. If the pixel is masked in all images, the filled color will be the average of the first sensor color of each pair

                When ``save_intermediate_data`` is activated, the folder ``dump_dir/auxiliary_filling`` will contain the non-filled color and classification.

                **Configuration**

                +------------------------------+-------------------------------------------+---------+----------------------------------+----------------------------------+----------+
                | Name                         | Description                               | Type    | Available values                 | Default value                    | Required |
                +==============================+===========================================+=========+==================================+==================================+==========+
                | method                       | Method for filling                        | string  | "auxiliary_filling_from_sensors" | "auxiliary_filling_from_sensors" | No       |
                +------------------------------+-------------------------------------------+---------+----------------------------------+----------------------------------+----------+
                | activated                    | Activates the filling                     | boolean |                                  | false                            | No       |
                +------------------------------+-------------------------------------------+---------+----------------------------------+----------------------------------+----------+
                | mode                         | Processing mode                           | string  | "fill_nan", "full"               | false                            | No       |
                +------------------------------+-------------------------------------------+---------+----------------------------------+----------------------------------+----------+
                | use_mask                     | Use mask information from input sensors   | boolean |                                  | true                             | No       |
                +------------------------------+-------------------------------------------+---------+----------------------------------+----------------------------------+----------+
                | color_interpolator           | interpolator used for color interpolation | string  | "linear", "nearest", "cubic"     | "linear"                         | No       |
                +------------------------------+-------------------------------------------+---------+----------------------------------+----------------------------------+----------+
                | save_intermediate_data       | Saves the temporary data in dump_dir      | boolean |                                  | false                            | No       |
                +------------------------------+-------------------------------------------+---------+----------------------------------+----------------------------------+----------+

    .. tab:: Advanced parameters

        Here are the advanced parameters. This key is optionnal and can be useful if you want to use CARS more as a developer.

        .. list-table:: Configuration
            :widths: 19 19 19 19 19
            :header-rows: 1

            * - Name
              - Description
              - Type
              - Default value
              - Required
            * - save_intermediate data
              - Save intermediate data for all applications
              - bool
              - False
              - Yes
            * - use_epipolar_a_priori
              - Active epipolar a priori
              - bool
              - False
              - Yes[Michel J. et al, 2020]
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
            * - performance_map_classes
              - List defining interval: [a,b,c,d] generates [[a,b],[b,c],[c,d]] intervals used in the performance map classification. If null, raw performance map is given
              - list or None
              - [0, 1.936, 2.2675, 2.59, 3.208, 4.846, 6.856]
              - No
            * - ground_truth_dsm
              - Datas to be reprojected from the application ground_truth_reprojection
              - dict
              -
              - No
            * - geometry_plugin
              - The plugin to use
              - str
              -
              - No
            * - pipeline
              - The pipeline to use
              - str
              -
              - No


        .. tabs::
	
            .. tab:: Save intermediate data

                The `save_intermediate_data` flag can be used to activate and deactivate the saving of the possible output of applications.

                It is set in the `advanced` category and can be overloaded in each application separately. It default to false, meaning that no intermediate product in saved). Intermediate data are saved in the `dump_dir` folder found in CARS output directory, with a subfolder corresponding to each application.

                For exemple setting `save_intermediate_data` to `true` in `advanced` and to `false` in `application/point_cloud_rasterization` will activate product saving in all applications excepting `point_cloud_rasterization`. Conversely, setting it to `false` in `advanced` and to `true` in `application/point_cloud_rasterization`  will only save rasterization outputs.

                Intermediate data refers to all files that are not part of an output product. Files that compose an output product will not be found in the application dump directory. For exemple if `dsm` is requested as output product, the `dsm.tif` files and all activated dsm auxiliary files will not be found in `rasterization` dump directory. This directory will still contain the files generated by the `rasterization` application that are not part of the `dsm` product.

                .. code-block:: json

                      "advanced": {
                          "save_intermediate_data": true
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
									"color": "path/to/color.tif"
								},
								"auxiliary_data_interpolation":{
									"classification": "nearest",
									"color": "linear"
								}
                            }
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
