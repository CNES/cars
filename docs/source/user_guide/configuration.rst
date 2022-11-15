
.. _configuration:

=============
Configuration
=============

This section describes main CARS configuration structure through a `json <http://www.json.org/json-fr.html>`_ configuration file.

The structure follows this organisation:

.. code-block:: json

    {
        "inputs": {},
        "orchestrator": {},
        "applications": {},
        "output": {}
    }
        
.. warning::

    Be careful with commas to separate each section. None needed for the last json element.

.. tabs::

   .. tab:: Inputs

    +---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
    | Name                | Description                                                         | Type                  | Default value        | Required |
    +=====================+=====================================================================+=======================+======================+==========+
    | *sensor*            | Stereo sensor images                                                | See next section      | No                   | Yes      |
    +---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
    | *pairing*           | Association of image to create pairs                                | list of *sensor*      | No                   | Yes      |
    +---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
    | *epsg*              | EPSG code                                                           | int, should be > 0    | None                 | No       |
    +---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
    | *initial_elevation* | Field contains the path to the folder in which are located          | string                | None                 | No       |
    |                     | the srtm tiles covering the production                              |                       |                      |          |
    +---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
    | *default_alt*       | Default height above ellipsoid when there is no DEM available       | int                   | 0                    | No       |
    |                     | no coverage for some points or pixels with no_data in the DEM tiles |                       |                      |          |
    +---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
    | *roi*               | DSM roi file or bounding box                                        | string, list or tuple | None                 | No       |
    +---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
    | *check_inputs*      | Check inputs consistency (to be deprecated and changed)             | Boolean               | False                | No       |
    +---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
    | *geoid*             | geoid path                                                          | string                | Cars internal geoid  | No       |
    +---------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+

    **Sensor**

    For each sensor images, give a particular name (what you want):

    .. code-block:: json

        {
            "my_name_for_this_image": 
            {
                "image" : "path_to_image.tif",
                "color" : "path_to_color.tif",
                "mask" : "path_to_mask.tif",
                "mask_classes" : {},
                "nodata": 0
            }
        }

    +-------------------+------------------------------------------------------------------------------------------+----------------+---------------+----------+
    | Name              | Description                                                                              | Type           | Default value | Required |
    +===================+==========================================================================================+================+===============+==========+
    | *image*           | Path to the image                                                                        | string         |               | Yes      |
    +-------------------+------------------------------------------------------------------------------------------+----------------+---------------+----------+
    | *color*           | image stackable to image used to create an ortho-image corresponding to the produced dsm | string         |               | No       |
    +-------------------+------------------------------------------------------------------------------------------+----------------+---------------+----------+
    | *no_data*         | no data value of the image                                                               | int            | -9999         | No       |
    +-------------------+------------------------------------------------------------------------------------------+----------------+---------------+----------+
    | *geomodel*        | geomodel associated to the image                                                         | string         |               | Yes      |
    +-------------------+------------------------------------------------------------------------------------------+----------------+---------------+----------+
    | *geomodel_filters*| filters associated to the geomodel                                                       | List of string |               | No       |
    +-------------------+------------------------------------------------------------------------------------------+----------------+---------------+----------+
    | *mask*            | external mask of the image                                                               | string         | None          | No       |
    +-------------------+------------------------------------------------------------------------------------------+----------------+---------------+----------+
    |*mask_classes*     | mask's classes usage (see next section for more details)                                 | dict           |               | No       |
    +-------------------+------------------------------------------------------------------------------------------+----------------+---------------+----------+

    .. note::

        - *color*: This image can be composed of XS bands in which case a PAN+XS fusion will be performed.
        - If the *mask* is a multi-classes one and no *mask_classes*  configuration file is indicated, all non-zeros values of the mask will be considered as invalid data.
        - The value 255 is reserved for CARS internal use, thus no class can be represented by this value in the masks.


    **CARS mask multi-classes structure**


    Multi-classes masks have a unified CARS format enabling the use of several mask information into the API.
    The classes can be used in different ways depending on the tag used in the dict defined below.

    Dict is given in the *mask_classes* fields of sensor (see previous section).
    This dict indicate the masks's classes usage and is structured as follows :

    .. code-block:: json

        {
            "ignored_by_dense_matching": [1, 2],
            "set_to_ref_alt": [1, 3, 4],
            "ignored_by_sparse_matching": [2]
        }


    * The classes listed in *ignored_by_sparse_matching* will be masked at the sparse matching step.
    * The classes listed in *ignored_by_dense_matching* will be masked at the dense matching step.
    * The classes listed in *set_to_ref_alt* will be set to the reference altitude (srtm or scalar). To do so, these pixels's disparity will be set to 0.


   .. tab:: Orchestrator

        CARS can distribute the computations chunks by using either dask (local or distributed cluster) or multiprocessing libraries.
        The distributed cluster require centralized files storage and uses PBS scheduler.

        The ``orchestrator`` key is optional and allows to define orchestrator configuration that controls the distribution:

        +------------------+---------------------------------------------------------------------+-----------------------------------------+---------------+----------+
        | Name             | Description                                                         | Type                                    | Default value | Required |
        +==================+=====================================================================+=========================================+===============+==========+
        | *mode*           | Parallelization mode "local_dask", "pbs_dask", "mp" or "sequential" | string                                  |local_dask     | Yes      |
        +------------------+---------------------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *profiling*      | Configuration for CARS profiling mode                               | dict                                    |               | No       |
        +------------------+---------------------------------------------------------------------+-----------------------------------------+---------------+----------+


        Depending on the used orchestrator mode, the following parameters can be added in the configuration:

        **Mode local_dask and pbs_dask:**

        +---------------------+-----------------------------------------------------------+-----------------------------------------+---------------+----------+
        | Name                | Description                                               | Type                                    | Default value | Required |
        +=====================+===========================================================+=========================================+===============+==========+
        | *nb_workers*        | Number of workers                                         | int, should be > 0                      | 2             | No       |
        +---------------------+-----------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *walltime*          | Walltime for one worker                                   | string, Should be formatted as HH:MM:SS | 00:59:00      | No       |
        +---------------------+-----------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *use_memory_logger* | Usage of dask memory logger                               | bool, True if use memory logger         | True          | No       |
        +---------------------+-----------------------------------------------------------+-----------------------------------------+---------------+----------+

        **Mode multiprocessing:**

        +---------------------+-----------------------------------------------------------+------------------------------------------+---------------+----------+
        | Name                | Description                                               | Type                                     | Default value | Required |
        +=====================+===========================================================+==========================================+===============+==========+
        | *nb_workers*        | Number of workers                                         | int, should be > 0                       | 2             | No       |
        +---------------------+-----------------------------------------------------------+------------------------------------------+---------------+----------+
        | *dump_to_disk*      | Dump temporary files to disk                              | bool, True if objects are dumped on disk | True          | No       |
        +---------------------+-----------------------------------------------------------+------------------------------------------+---------------+----------+
        | *per_job_timeout*   | Timeout used for a job                                    | float, int                               | 600           | No       |
        +---------------------+-----------------------------------------------------------+------------------------------------------+---------------+----------+
        | *factorize_delayed* | Factorize delayed tasks                                   | bool                                     | True          | No       |
        +---------------------+-----------------------------------------------------------+------------------------------------------+---------------+----------+

        **Profiling configuration:**

        The profiling mode is used to analyze time or memory of the executed CARS functions at worker level. By default, the profiling mode is disabled.
        It could be configured for the different orchestrator modes and for different purposes (time, elapsed time, memory allocation, loop testing).

        .. code-block:: json

            {
                "orchestrator": 
                {
                    "mode" : "sequential",
                    "profiling" : {},
                }
            }

        +---------------------+-----------------------------------------------------------+-----------------------------------------+---------------+----------+
        | Name                | Description                                               | Type                                    | Default value | Required |
        +=====================+===========================================================+=========================================+===============+==========+
        | *activated*         | activation of the profiling mode (disabled by default)    | bool                                    | False         | No       |
        +---------------------+-----------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *mode*              | type of profiling mode "time, cprofile, memray"           | string                                  | time          | No       |
        +---------------------+-----------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *loop_testing*      | enable loop mode to execute each step multiple times      | bool                                    | False         | No       |
        +---------------------+-----------------------------------------------------------+-----------------------------------------+---------------+----------+

        - Please use make command 'profile-memory-report' to generate a memory profiling report from the memray outputs files (after the memray profiling execution).
        - Please disabled profiling to eval memory profiling at master orchestrator level and execute make command instead: 'profile-memory-all'.

   .. tab:: Applications

    This key is optional and allows to redefine parameters for each application used in pipeline as described in :ref:`overview`

    This section describes all possible configuration of CARS applications.

    CARS applications are defined and called by their name in applications configuration section:

    .. code-block:: json

      "applications":{
          "application_name": {
              "method": "application_dependent",
              "parameter1": 3,
              "parameter2": 0.3
          }
      },

    Be careful with these parameters: no mechanism ensures consistency between applications for now.
    And some parameters can degrade performance and DSM quality heavily.
    The default parameters have been set as a robust and consistent end to end configuration for the whole pipeline.

    .. tabs::

        .. tab:: Grid Generation

            **Name**: "grid_generation"

            **Description**

            From sensors image, compute the stereo-rectification grids

            **Configuration**

            +-----------------+-----------------------------------------------+---------+---------------+----------+
            | Name            | Description                                   | Type    | Default value | Required |
            +=================+===============================================+=========+===============+==========+
            | method          | Method for grid generation                    | string  | epipolar      | Yes      |
            +-----------------+-----------------------------------------------+---------+---------------+----------+
            | epi_step        | Step of the deformation grid in nb. of pixels | int     | 30            | No       |
            +-----------------+-----------------------------------------------+---------+---------------+----------+
            | save_grids      | Save the generated grids (not available yet)  | boolean | false         | No       |
            +-----------------+-----------------------------------------------+---------+---------------+----------+
            | geometry_loader | Geometry external library                     | string  | "otb"         | No       |
            +-----------------+-----------------------------------------------+---------+---------------+----------+

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

            +---------------------+--------------------------------------------------------+---------+---------------+----------+
            | Name                | Description                                            | Type    | Default value | Required |
            +=====================+========================================================+=========+===============+==========+
            | method              | Method for resampling                                  | string  | bicubic       | Yes      |
            +---------------------+--------------------------------------------------------+---------+---------------+----------+
            | epi_tile_size       | size in pixels of tile                                 | int     | 500           | No       |
            +---------------------+--------------------------------------------------------+---------+---------------+----------+
            | save_epipolar_image | Save the generated images in output folder             | boolean | false         | No       |
            +---------------------+--------------------------------------------------------+---------+---------------+----------+
            | save_epipolar_color | Save the generated images (only if color is available) | boolean | false         | No       |
            +---------------------+--------------------------------------------------------+---------+---------------+----------+

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

            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+
            | Name                                 | Description                                                                                 | Type       | available value | Default value | Required |
            +======================================+=============================================================================================+============+=================+===============+==========+
            | method                               | Method for sparse matching                                                                  | string     | "sift"          | "sift"        | Yes      |
            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+
            | disparity_margin                     | Add a margin to min and max disparity as percent of the disparity range.                    | float      |                 | 0.02          | No       |
            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+
            | elevation_delta_lower_bound          | Expected lower bound for elevation delta with respect to input low resolution DTM in meters | int, float |                 | -1000         | No       |
            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+
            | elevation_delta_upper_bound          | Expected upper bound for elevation delta with respect to input low resolution DTM in meters | int, float |                 | 1000          | No       |
            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+
            | epipolar_error_upper_bound           | Expected upper bound for epipolar error in pixels                                           | float      |                 | 10.0          | No       |
            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+
            | epipolar_error_maximum_bias          | Maximum bias for epipolar error in pixels                                                   | float      |                 | 0.0           | No       |
            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+
            | disparity_outliers_rejection_percent | Percentage of outliers to reject                                                            | float      |                 | 0.1           | No       |
            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+
            | minimum_nb_matches                   | Minimum number of matches that must be computed to continue pipeline                        | int        |                 | 100           | No       |
            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+
            | sift_matching_threshold              |                                                                                             | float      |                 | 0.6           | No       |
            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+
            | sift_n_octave                        |                                                                                             | int        |                 | 8             | No       |
            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+
            | sift_n_scale_per_octave              |                                                                                             | int        |                 | 3             | No       |
            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+
            | sift_dog_threshold                   |                                                                                             | float      |                 | 20.0          | No       |
            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+
            | sift_edge_threshold                  |                                                                                             | float      |                 | -5.0          | No       |
            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+
            | sift_magnification                   |                                                                                             | float      |                 | 2.0           | No       |
            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+
            | sift_back_matching                   |                                                                                             | Boolean    |                 | true          | No       |
            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+
            | save_matches                         | Save matches                                                                                | Boolean    |                 | false         | No       |
            +--------------------------------------+---------------------------------------------------------------------------------------------+------------+-----------------+---------------+----------+

            A lot of information about parameters can be found on `VLFEAT SIFT documentation <https://www.vlfeat.org/api/sift.html>`_.

            **Example**

            .. code-block:: json

                "applications": {
                    "sparse_matching": {
                        "method": "sift",
                        "disparity_margin": 0.01
                    }
                },

        .. tab:: Dense matching

            **Name**: "dense_matching"

            **Description**

            Compute disparity map from stereo-rectified pair images

            **Configuration**

            +---------------------------------+-------------------------------------------------------------------------+---------+---------------------------------+---------------+----------+
            | Name                            | Description                                                             | Type    | available value                 | Default value | Required |
            +---------------------------------+-------------------------------------------------------------------------+---------+---------------------------------+---------------+----------+
            | method                          | Method for dense matching                                               | string  | "census_sgm" or "mccnn_sgm"     | "census_sgm"  | Yes      |
            +---------------------------------+-------------------------------------------------------------------------+---------+---------------------------------+---------------+----------+
            | loader                          | external library use to compute dense matching                          | string  | "pandora"                       | "pandora"     | No       |
            +---------------------------------+-------------------------------------------------------------------------+---------+---------------------------------+---------------+----------+
            | loader_conf                     | Configuration associated with loader                                    | dict    |                                 |               | No       |
            +---------------------------------+-------------------------------------------------------------------------+---------+---------------------------------+---------------+----------+
            | min_elevation_offset            | Override minimum disparity from prepare step with this offset in meters | int     |                                 | None          | No       |
            +---------------------------------+-------------------------------------------------------------------------+---------+---------------------------------+---------------+----------+
            | max_elevation_offset            | Override maximum disparity from prepare step with this offset in meters | int     |                                 | None          | No       |
            +---------------------------------+-------------------------------------------------------------------------+---------+---------------------------------+---------------+----------+
            | use_sec_disp                    | Compute secondary disparity map                                         | boolean |                                 | false         | No       |
            +---------------------------------+-------------------------------------------------------------------------+---------+---------------------------------+---------------+----------+
            | min_epi_tile_size               |                                                                         | int     |                                 | 300           | No       |
            +---------------------------------+-------------------------------------------------------------------------+---------+---------------------------------+---------------+----------+
            | max_epi_tile_size               |                                                                         | int     |                                 | 300           | No       |
            +---------------------------------+-------------------------------------------------------------------------+---------+---------------------------------+---------------+----------+
            | epipolar_tile_margin_in_percent |                                                                         | int     |                                 | 60            | No       |
            +---------------------------------+-------------------------------------------------------------------------+---------+---------------------------------+---------------+----------+
            | save_disparity_map              | Save disparity map and ambiguity confidence                             | boolean |                                 | false         | No       |
            +---------------------------------+-------------------------------------------------------------------------+---------+---------------------------------+---------------+----------+

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

        .. tab:: HolesDetection

            **Name**: "holes_detection"

            **Description**

            Detection of holes from input maskes.

            **Configuration**

            +-------------------+-------------------------------+---------+------------------------------+------------------------------+----------+
            | Name              | Description                   | Type    | available value              | Default value                | Required |
            +===================+===============================+=========+==============================+==============================+==========+
            | method            | Method for holes detection    | string  | "cloud_to_bbox"              | "cloud_to_bbox"              | Yes      |
            +-------------------+-------------------------------+---------+------------------------------+------------------------------+----------+

            **Example**

            .. code-block:: json

                "applications": {
                    "holes_detection": {
                        "method": "cloud_to_bbox"
                    }
                },

            .. warning::

              This application will not be used if DenseMatchingFiling activated parameter is false

        .. tab:: DenseMatchingFiling

            **Name**: "dense_matches_filling"

            **Description**

            Fill holes in dense matches map. This uses the holes detected with the HolesDetection application.

            **Configuration**

            +-------------------------------------+---------------------------------+---------+---------------------+--------------------+----------+
            | Name                                | Description                     | Type    | available value     | Default value      | Required |
            +=====================================+=================================+=========+=====================+====================+==========+
            | method                              | Method for holes detection      | string  | "plane"             | "plane"            | Yes      |
            +-------------------------------------+---------------------------------+---------+---------------------+--------------------+----------+
            | save_disparity_map                  | Save disparity map              | boolean |                     |False               | No       |
            +-------------------------------------+---------------------------------+---------+---------------------+--------------------+----------+
            | activated                           | Activate application            | boolean |                     | false              | No       |
            +-------------------------------------+---------------------------------+---------+---------------------+--------------------+----------+
            | interpolation_type                  | Interpolation type              | string  | "pandora"           | "pandora"          | No       |
            +-------------------------------------+---------------------------------+---------+---------------------+--------------------+----------+
            | interpolation_method                | Method for holes interpolation  | string  | "mc_cnn"            | "mc_cnn"           | No       |
            +-------------------------------------+---------------------------------+---------+---------------------+--------------------+----------+
            | max_search_distance                 | Maximum search distance         | int     |                     | 100                | No       |
            +-------------------------------------+---------------------------------+---------+---------------------+--------------------+----------+
            | smoothing_iterations                | Number of smoothing iterations  | int     |                     | 1                  | No       |
            +-------------------------------------+---------------------------------+---------+---------------------+--------------------+----------+
            | ignore_nodata_at_disp_mask_borders  | Ingnore nodata at borders       | boolean |                     | true               | No      |
            +-------------------------------------+---------------------------------+---------+---------------------+--------------------+----------+
            | ignore_zero_fill_disp_mask_values   | Ignore zeros                    | boolean |                     | true               | No       |
            +-------------------------------------+---------------------------------+---------+---------------------+--------------------+----------+
            | ignore_extrema_disp_values          | Ignore extrema values           | boolean |                     | true               | No       |
            +-------------------------------------+---------------------------------+---------+---------------------+--------------------+----------+
            | nb_pix                              | Margin used for mask            | int     |                     | 20                 | No       |
            +-------------------------------------+---------------------------------+---------+---------------------+--------------------+----------+
            | percent_to_erode                    | Percentage to erode             | float   |                     | 0.2                | No       |
            +-------------------------------------+---------------------------------+---------+---------------------+--------------------+----------+

            **Example**

            .. code-block:: json

                "applications": {
                    "dense_matches_filling": {
                        "method": "plane",
                        "activated": true,
                        "save_disparity_map": true
                    }
                },
            
            

        .. tab:: Triangulation

            **Name**: "triangulation"

            **Description**

            Triangulating the sights and get for each point of the reference image a latitude, longitude, altitude point

            **Configuration**

            +-------------------+--------------------------------------------------------------------------------------------------------------------+---------+------------------------------+------------------------------+----------+
            | Name              | Description                                                                                                        | Type    | available value              | Default value                | Required |
            +===================+====================================================================================================================+=========+==============================+==============================+==========+
            | method            | Method for triangulation                                                                                           | string  | "line_of_sight_intersection" | "line_of_sight_intersection" | Yes      |
            +-------------------+--------------------------------------------------------------------------------------------------------------------+---------+------------------------------+------------------------------+----------+
            | geometry_loader   | Geometry external library                                                                                          | string  | "otb"                        | "otb"                        | No       |
            +-------------------+--------------------------------------------------------------------------------------------------------------------+---------+------------------------------+------------------------------+----------+
            | use_geoid_alt     | Use geoid grid as altimetric reference.                                                                            | boolean |                              | false                        | No       |
            +-------------------+--------------------------------------------------------------------------------------------------------------------+---------+------------------------------+------------------------------+----------+
            | snap_to_img1      | if all pairs share the same left image, modify lines of sights of secondary images to cross those of the ref image | boolean |                              | false                        | No       |
            +-------------------+--------------------------------------------------------------------------------------------------------------------+---------+------------------------------+------------------------------+----------+
            | add_msk_info      |                                                                                                                    | boolean |                              | true                         | No       |
            +-------------------+--------------------------------------------------------------------------------------------------------------------+---------+------------------------------+------------------------------+----------+
            | save_points_cloud | save points_cloud                                                                                                  | boolean |                              | false                        | No       |
            +-------------------+--------------------------------------------------------------------------------------------------------------------+---------+------------------------------+------------------------------+----------+

            **Example**

            .. code-block:: json

                "applications": {
                    "triangulation": {
                        "method": "line_of_sight_intersection",
                        "use_geoid_alt": true
                    }
                },

        .. tab:: Point Cloud fusion

            **Name**: "point_cloud_fusion"

            **Description**

            Merge points clouds coming from each pair

            Only one method is available for now: "mapping_to_terrain_tiles"

            **Configuration**

            +--------------------------+----------------------------------+---------+----------------------------+----------------------------+----------+
            | Name                     | Description                      | Type    | available value            | Default value              | Required |
            +==========================+==================================+=========+============================+============================+==========+
            | method                   | Method for fusion                | string  | "mapping_to_terrain_tiles" | "mapping_to_terrain_tiles" | Yes      |
            +--------------------------+----------------------------------+---------+----------------------------+----------------------------+----------+
            | resolution               | Resolution of the dsm            | float   | should be > 0              | 0.5                        | No       |
            +--------------------------+----------------------------------+---------+----------------------------+----------------------------+----------+
            | terrain_tile_size        |                                  | int     |                            | None                       | No       |
            +--------------------------+----------------------------------+---------+----------------------------+----------------------------+----------+
            | save_points_cloud_as_laz | Save points clouds as laz format | boolean |                            | false                      | No       |
            +--------------------------+----------------------------------+---------+----------------------------+----------------------------+----------+
            | save_points_cloud_as_csv | Save points clouds as csv format | boolean |                            | false                      | No       |
            +--------------------------+----------------------------------+---------+----------------------------+----------------------------+----------+

            **Example**


            .. code-block:: json

                    "applications": {
                        "point_cloud_fusion": {
                            "method": "mapping_to_terrain_tiles",
                            "resolution": 0.5,
                            "save_points_cloud_as_laz": true,
                            "save_points_cloud_as_csv": true,
                        }
                    },

            .. warning::

              Be careful with resolution to be consistent with resolution in rasterization.
              No mechanism ensures consistency between applications for now.


        .. tab:: Point Cloud outliers removing

            **Name**: "point_cloud_outliers_removing"

            **Description**

            Point cloud outliers removing

            **Configuration**

            +--------------------------+------------------------------------------+---------+-----------------------------------+---------------+----------+
            | Name                     | Description                              | Type    | available value                   | Default value | Required |
            +==========================+==========================================+=========+===================================+===============+==========+
            | method                   | Method for point cloud outliers removing | string  | "statistical", "small_components" | "statistical" | Yes      |
            +--------------------------+------------------------------------------+---------+-----------------------------------+---------------+----------+
            | save_points_cloud_as_laz | Save points clouds as laz format         | boolean |                                   | false         | No       |
            +--------------------------+------------------------------------------+---------+-----------------------------------+---------------+----------+
            | save_points_cloud_as_csv | Save points clouds as csv format         | boolean |                                   | false         | No       |
            +--------------------------+------------------------------------------+---------+-----------------------------------+---------------+----------+

            If method is *statistical*:

            +----------------+-------------+---------+-----------------+---------------+----------+
            | Name           | Description | Type    | available value | Default value | Required |
            +================+=============+=========+=================+===============+==========+
            | activated      |             | boolean |                 | true          | No       |
            +----------------+-------------+---------+-----------------+---------------+----------+
            | k              |             | int     | should be > 0   | 50            | No       |
            +----------------+-------------+---------+-----------------+---------------+----------+
            | std_dev_factor |             | float   |                 | 5.0           | No       |
            +----------------+-------------+---------+-----------------+---------------+----------+

            If method is *small_components*

            +-----------------------------+-------------+---------+-----------------+---------------+----------+
            | Name                        | Description | Type    | available value | Default value | Required |
            +=============================+=============+=========+=================+===============+==========+
            | activated                   |             | boolean |                 | true          | No       |
            +-----------------------------+-------------+---------+-----------------+---------------+----------+
            | on_ground_margin            |             | int     |                 | 10            | No       |
            +-----------------------------+-------------+---------+-----------------+---------------+----------+
            | connection_distance         |             | float   |                 | 3.0           | No       |
            +-----------------------------+-------------+---------+-----------------+---------------+----------+
            | nb_points_threshold         |             | int     |                 | 50            | No       |
            +-----------------------------+-------------+---------+-----------------+---------------+----------+
            | clusters_distance_threshold |             | float   |                 | None          | No       |
            +-----------------------------+-------------+---------+-----------------+---------------+----------+

            .. warning::

                There is a particular case with the *Point Cloud outliers removing* application because it is called twice.
                As described on :ref:`overview`, the ninth step consists of Filter the 3D points cloud via two consecutive filters.
                So you can configure the application twice , once for the *small component filters*, the other for *statistical* filter.
                Because it is not possible to define twice the *application_name* on your json configuration file, we have decided to configure
                those two applications with :

                 * *point_cloud_outliers_removing.1*
                 * *point_cloud_outliers_removing.2*

                Each one is associated to a particular *point_cloud_outliers_removing* method*

            **Example**

            .. code-block:: json

                    "applications": {
                        "point_cloud_outliers_removing.1": {
                            "method": "small_components",
                            "on_ground_margin": 10,
                            "save_points_cloud_as_laz": true,
                            "save_points_cloud_as_csv": false
                        },
                        "point_cloud_outliers_removing.2": {
                            "method": "statistical",
                            "k": 10,
                            "save_points_cloud_as_laz": true,
                        }
                    },

        .. tab:: Point Cloud Rasterization

            **Name**: "point_cloud_rasterization"

            **Description**

            Project altitudes on regular grid.

            Only one simple gaussian method is available for now.

            **Configuration**

            +-----------------------------+------------------------+------------+-----------------+-----------------+----------+
            | Name                        | Description            | Type       | available value | Default value   | Required |
            +=============================+========================+============+=================+=================+==========+
            | method                      |                        | string     | simple_gaussian | simple_gaussian | Yes      |
            +-----------------------------+------------------------+------------+-----------------+-----------------+----------+
            | dsm_radius                  |                        | float, int |                 | 1.0             | No       |
            +-----------------------------+------------------------+------------+-----------------+-----------------+----------+
            | sigma                       |                        | float      |                 | None            | No       |
            +-----------------------------+------------------------+------------+-----------------+-----------------+----------+
            | grid_points_division_factor |                        | int        |                 | None            | No       |
            +-----------------------------+------------------------+------------+-----------------+-----------------+----------+
            | resolution                  |altitude grid step (dsm)| float      |                 | 0.5             | No       |
            +-----------------------------+------------------------+------------+-----------------+-----------------+----------+
            | dsm_no_data                 |                        | int        |                 | -32768          |          |
            +-----------------------------+------------------------+------------+-----------------+-----------------+----------+
            | color_no_data               |                        | int        |                 | 0               |          |
            +-----------------------------+------------------------+------------+-----------------+-----------------+----------+
            | color_dtype                 |                        | string     |                 | "uint16"        |          |
            +-----------------------------+------------------------+------------+-----------------+-----------------+----------+
            | msk_no_data                 |                        | int        |                 | 65535           |          |
            +-----------------------------+------------------------+------------+-----------------+-----------------+----------+
            | write_color                 | Save color ortho-image | boolean    |                 | false           | No       |
            +-----------------------------+------------------------+------------+-----------------+-----------------+----------+
            | write_stats                 |                        | boolean    |                 | false           | No       |
            +-----------------------------+------------------------+------------+-----------------+-----------------+----------+
            | write_msk                   |                        | boolean    |                 | false           | No       |
            +-----------------------------+------------------------+------------+-----------------+-----------------+----------+
            | write_dsm                   | Save dsm               | boolean    |                 | true            | No       |
            +-----------------------------+------------------------+------------+-----------------+-----------------+----------+
            | write_ambiguity             | Save ambiguity         | boolean    |                 | true            | No       |
            +-----------------------------+------------------------+------------+-----------------+-----------------+----------+
            | compute_all                 | Compute all layers even| boolean    |                 | false           | No       |
            |                             | if one or more layers  |            |                 |                 |          |
            |                             | are not saved (color   |            |                 |                 |          |
            |                             | , stats, dsm, msk..)   |            |                 |                 |          |
            +-----------------------------+------------------------+------------+-----------------+-----------------+----------+

            **Example**

            .. code-block:: json

                    "applications": {
                        "point_cloud_rasterization": {
                            "method": "simple_gaussian",
                            "dsm_radius": 1.5
                        }
                    },

            .. warning::

                Be careful with resolution to be consistent with resolution in rasterization.
                No mechanism ensures consistency between applications for now.

   .. tab:: Outputs

        +----------------+-------------------------------------------------------------+--------+----------------+----------+
        | Name           | Description                                                 | Type   | Default value  | Required |
        +================+=============================================================+========+================+==========+
        | out_dir        | Output folder where results are stored                      | string | No             | Yes      |
        +----------------+-------------------------------------------------------------+--------+----------------+----------+
        | dsm_basename   | base name for dsm                                           | string | "dsm.tif"      | No       |
        +----------------+-------------------------------------------------------------+--------+----------------+----------+
        | color_basename | base name for  ortho-image                                  | string | "color.tif     | No       |
        +----------------+-------------------------------------------------------------+--------+----------------+----------+
        | info_basename  | base name for file containing information about computation | string | "content.json" | No       |
        +----------------+-------------------------------------------------------------+--------+----------------+----------+

        *Output contents*

        The output directory, defined on the configuration file (see previous section) contains at the end of the computation:

        * the dsm
        * color image (if *color image* has been given)
        * information json file containing: used parameters, information and numerical results related to computation, step by step and pair by pair.
        * subfolder for each defined pair which can contains intermediate data



Full example
============

Here is a full detailed example with **orchestrator** and **applications** capabilities. See correspondent sections for details.

.. code-block:: json

    {
      "inputs": {
          "sensors" : {
              "one": {
                  "image": "img1.tif",
                  "geomodel": "img1.geom",
                  "no_data": 0
              },
              "two": {
                  "image": "img2.tif",
                  "geomodel": "img2.geom",
                  "no_data": 0

              },
              "three": {
                  "image": "img3.tif",
                  "geomodel": "img3.geom",
                  "no_data": 0
              }
          },
          "pairing": [["one", "two"],["one", "three"]],
          "initial_elevation": "srtm_dir"
        },

        "orchestrator": {
            "mode":"local_dask",
            "nb_workers": 4
        },

        "applications":{
            "point_cloud_rasterization": {
                "method": "simple_gaussian",
                "dsm_radius": 3,
                "sigma": 0.3
            }
        },

        "output": {
          "out_dir": "outresults"
        }
      }





