.. _basic configuration:

Basic configuration
===================

This section describes CARS main basic configuration structure through a `json <http://www.json.org/json-fr.html>`_ configuration file. See how to launch it in section :ref:`How to launch CARS`.


The structure follows this organization:

.. code-block:: json

    {
        "inputs": {},
        "orchestrator": {},
        "output": {},
    }

.. warning::

    Be careful with commas to separate each section. None needed for the last json element.

.. tabs::

    .. tab:: Inputs

        CARS can be entered either with Sensor Images or with Depth Maps. 
        
        Additional inputs can be provided for both types of inputs, namely a ROI and an initial elevation.

        .. tabs::

            .. tab:: Sensors Images inputs

                +----------------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
                | Name                       | Description                                                         | Type                  | Default value        | Required |
                +============================+=====================================================================+=======================+======================+==========+
                | *sensors*                  | Stereo sensor images                                                | See next section      | No                   | Yes      |
                +----------------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
                | *pairing*                  | Association of image to create pairs                                | list of *sensors*     | No                   | Yes (*)  |
                +----------------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+

                (*) `pairing` is required if there are more than two sensors (see pairing section below)

                **Sensor**

                For each sensor image, give a particular name (what you want):

                .. code-block:: json

                    {
                        "inputs": {
                            "sensors": {
                                "my_name_for_this_image": {
                                    "image" : "path_to_image.tif",
                                    "color" : "path_to_color.tif",
                                    "mask" : "path_to_mask.tif",
                                    "classification" : "path_to_classification.tif",
                                    "nodata": 0
                                }
                            }
                        }
                    }

                +-------------------+--------------------------------------------------------------------------------------------+----------------+---------------+----------+
                | Name              | Description                                                                                | Type           | Default value | Required |
                +===================+============================================================================================+================+===============+==========+
                | *image*           | Path to the image                                                                          | string         |               | Yes      |
                +-------------------+--------------------------------------------------------------------------------------------+----------------+---------------+----------+
                | *color*           | Path to the color image                                                                    | string         |               | No       |
                +-------------------+--------------------------------------------------------------------------------------------+----------------+---------------+----------+
                | *no_data*         | No data value of the image                                                                 | int            | 0             | No       |
                +-------------------+--------------------------------------------------------------------------------------------+----------------+---------------+----------+
                | *geomodel*        | Path to the geomodel and plugin-specific attributes                                        | string, dict   |               | No       |
                +-------------------+--------------------------------------------------------------------------------------------+----------------+---------------+----------+
                | *mask*            | Path to the binary mask                                                                    | string         | None          | No       |
                +-------------------+--------------------------------------------------------------------------------------------+----------------+---------------+----------+
                | *classification*  | Path to the multiband binary classification image                                          | string         | None          | No       |
                +-------------------+--------------------------------------------------------------------------------------------+----------------+---------------+----------+

                .. note::

                    - *color*: This image can be composed of XS bands in which case a PAN+XS fusion has been be performed. Please, see the section :ref:`make_a_simple_pan_sharpening` to make a simple pan sharpening with OTB if necessary.
                    - *mask*: This image is a binary file. By using this file, the 1 values are not processed, only 0 values are considered as valid data.
                    - *classification*: This image is a multiband binary file. Each band should have a specific name (Please, see the section :ref:`add_band_description_in_image` to add band name / description in order to be used in Applications). By using this file, a different process for each band is applied for the 1 values (Please, see the Applications section for details).
                    - Please, see the section :ref:`convert_image_to_binary_image` to make binary *mask* image or binary *classification* image with 1 bit per band.
                    - *geomodel*: If the geomodel file is not provided, CARS will try to use the RPC loaded with rasterio opening *image*.
                    - It is possible to add sensors inputs while using depth_maps or dsm inputs

                **Pairing**

                The `pairing` attribute defines the pairs to use, using sensors keys used to define sensor images.

                .. code-block:: json

                    {
                        "inputs": {
                            "sensors" : {
                                "one": {
                                    "image": "img1.tif",
                                    "geomodel": "img1.geom"
                                },
                                "two": {
                                    "image": "img2.tif",
                                    "geomodel": "img2.geom"

                                },
                                "three": {
                                    "image": "img3.tif",
                                    "geomodel": "img3.geom"
                                }
                            },
                            "pairing": [["one", "two"],["one", "three"]]
                        }
                    }

                This attribute is required when there are more than two input sensor images. If only two images ares provided, the pairing can be deduced by cars, considering the first image defined as the left image and second image as right image.


            .. tab:: Depth Maps inputs

                +-------------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
                | Name                    | Description                                                         | Type                  | Default value        | Required |
                +=========================+=====================================================================+=======================+======================+==========+
                | *depth_maps*            | Depth maps to rasterize                                             | dict                  | No                   | Yes      |
                +-------------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+



                **Depth Maps**

                For each depth map, give a particular name (what you want):

                .. code-block:: json

                    {
                        "inputs": {
                            "depth_maps": {
                                "my_name_for_this_depth_map": {
                                    "x" : "path_to_x.tif",
                                    "y" : "path_to_y.tif",
                                    "z" : "path_to_z.tif",
                                    "color" : "path_to_color.tif",
                                    "z_inf" : "path_to_z_inf.tif",
                                    "z_sup" : "path_to_z_sup.tif",
                                    "mask": "path_to_mask.tif",
                                    "classification": "path_to_classification.tif",
                                    "filling": "path_to_filling.tif",
                                    "confidence": {
                                        "confidence_name1": "path_to_confidence1.tif",
                                        "confidence_name2": "path_to_confidence2.tif",
                                    },
                                    "performance_map": "path_to_performance_map.tif",
                                    "epsg": "depth_map_epsg"
                                }
                            }
                        }
                    }

                These input files can be generated by running CARS with `product_level: ["depth_map"]` and `auxiliary` dictionary filled with desired auxiliary files

                .. note::

                    To generate confidence maps, `z_inf` and `z_sup`, the parameter `save_intermediate_data` of `triangulation` should be activated.

                    To generate the performance map, the parameters `generate_performance_map` and `save_intermediate_data` of the `dense_matching` application must be activated.

                    It is possible to add sensors inputs while using depth_maps inputs
                    
                +------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | Name             | Description                                                       | Type           | Default value | Required |
                +==================+===================================================================+================+===============+==========+
                | *x*              | Path to the x coordinates of depth map                            | string         |               | Yes      |
                +------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *y*              | Path to the y coordinates of depth map                            | string         |               | Yes      |
                +------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *z*              | Path to the z coordinates of depth map                            | string         |               | Yes      |
                +------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *color*          | Color of depth map                                                | string         |               | Yes      |
                +------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *z_inf*          | Path to the z_inf coordinates of depth map                        | string         |               | No       |
                +------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *z_sup*          | Path to the z_sup coordinates of depth map                        | string         |               | No       |
                +------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *mask*           | Validity mask of depth map   : 0 values are considered valid data | string         |               | No       |
                +------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *classification* | Classification of depth map                                       | string         |               | No       |
                +------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *filling*        | Filling map of depth map                                          | string         |               | No       |
                +------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *confidence*     | Dict of paths to the confidences of depth map                     | dict           |               | No       |
                +------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *epsg*           | Epsg code of depth map                                            | int            | 4326          | No       |
                +------------------+-------------------------------------------------------------------+----------------+---------------+----------+

            .. tab:: DSMS inputs

                +-------------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
                | Name                    | Description                                                         | Type                  | Default value        | Required |
                +=========================+=====================================================================+=======================+======================+==========+
                | *dsm*                   | Dsms to merge                                                       | dict                  | No                   | Yes      |
                +-------------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+



                **DSMS**

                For each DSMS, give a particular name (what you want):

                .. code-block:: json

                    {
                        "inputs": {
                            "dsms": {
                                "my_name_for_this_dsm": {
                                    "dsm" : "path_to_dsm.tif",
                                    "classification" : "path_to_classif.tif",
                                    "color" : "path_to_color.tif",
                                    "performance_map" : "path_to_performance_map.tif",
                                    "filling" : "path_to_filling.tif",
                                    "mask" : "path_to_mask.tif",
                                    "weights": "path_to_weights.tif",
                                    "dsm_inf": "path_to_dsm_inf.tif",
                                    "dsm_sup": "path_to_dsm_sup.tif"
                                }
                            }
                        }
                    }

                These input files can be generated by running CARS with `product_level: ["dsm"]` and `auxiliary` dictionary filled with desired auxiliary files

                .. note::

                    To generate confidence maps, `z_inf` and `z_sup`, the parameter `save_intermediate_data` of `triangulation` should be activated.

                    To generate the performance map, the parameters `generate_performance_map` and `save_intermediate_data` of the `dense_matching` application must be activated.

                    It is possible to add sensors inputs while using dsm inputs
                    
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | Name                       | Description                                                       | Type           | Default value | Required |
                +============================+===================================================================+================+===============+==========+
                | *dsm*                      | Path to the dsm file                                              | string         |               | Yes      |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *weights*                  | Path to the weights file                                          | string         |               | Yes      |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *color*                    | Path to the color file                                            | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *classification*           | Path to the classification file                                   | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *mask*                     | Path to the mask file                                             | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *filling*                  | Path to the filling file                                          | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *performance_map*          | Path to the performance_map file                                  | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *source_pc*                | Path to the source_pc file                                        | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *dsm_inf*                  | Path to the dsm_inf file                                          | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *dsm_sup*                  | Path to the dsm_sup file                                          | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *dsm_mean*                 | Path to the dsm_mean file                                         | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
        	| *dsm_std*                  | Path to the dsm_std file                                          | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *dsm_inf_mean*             | Path to the dsm_inf_mean file                                     | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
        	| *dsm_inf_std*              | Path to the dsm_inf_std file                                      | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *dsm_sup_mean*             | Path to the dsm_sup_mean file                                     | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *dsm_sup_std*              | Path to the dsm_sup_std file                                      | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
        	| *dsm_n_pts*                | Path to the dsm_n_pts file                                        | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *dsm_pts_in_cell*          | Path to the dsm_pts_in_cell file                                  | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
        	| *confidence_from_ambiguity*| Path to the confidence_from_ambiguity file                        | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *confidence_from_risk_min* | Path to the confidence_from_risk_min file                         | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+
                | *confidence_from_risk_max* | Path to the confidence_from_risk_min file                         | string         |               | No       |
                +----------------------------+-------------------------------------------------------------------+----------------+---------------+----------+

            .. tab:: ROI

                +-------------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
                | Name                    | Description                                                         | Type                  | Default value        | Required |
                +=========================+=====================================================================+=======================+======================+==========+
                | *roi*                   | Region Of Interest: Vector file path or GeoJson dictionary          | string, dict          | None                 | No       |
                +-------------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+

                **ROI**

                A terrain ROI can be provided by the user. It can be either a vector file (Shapefile for instance) path,
                or a GeoJson dictionary. These structures must contain a single Polygon or MultiPolygon. Multi-features are 
                not supported. Instead of cropping the input images, the whole images will be used to compute grid correction
                and terrain + epipolar a priori. Then the rest of the pipeline will use the given roi. T
                his allow better correction of epipolar rectification grids.


                Example of the "roi" parameter with a GeoJson dictionary containing a Polygon as feature :

                .. code-block:: json

                    {
                        "inputs":
                        {
                            "roi" : {
                                "type": "FeatureCollection",
                                "features": [
                                    {
                                    "type": "Feature",
                                    "properties": {},
                                    "geometry": {
                                        "coordinates": [
                                        [
                                            [5.194, 44.2064],
                                            [5.194, 44.2059],
                                            [5.195, 44.2059],
                                            [5.195, 44.2064],
                                            [5.194, 44.2064]
                                        ]
                                        ],
                                        "type": "Polygon"
                                    }
                                    }
                                ]
                            }
                        }
                    }

                If the *debug_with_roi* advanced parameter (see dedicated tab) is enabled, the tiling of the entire image is kept but only the tiles intersecting 
                the ROI are computed.

                MultiPolygon feature is only useful if the parameter *debug_with_roi* is activated, otherwise the total footprint of the 
                MultiPolygon will be used as ROI. 

                By default epsg 4326 is used. If the user has defined a polygon in a different reference system, the "crs" field must be specified.

                Example of the *debug_with_roi* mode utilizing an "roi" parameter of type MultiPolygon as a feature and a specific EPSG.

                .. code-block:: json

                    {
                        "inputs":
                        {
                            "roi" : {
                                "type": "FeatureCollection",
                                "features": [
                                    {
                                    "type": "Feature",
                                    "properties": {},
                                    "geometry": {
                                        "coordinates": [
                                        [
                                            [
                                                [319700, 3317700],
                                                [319800, 3317700],
                                                [319800, 3317800],
                                                [319800, 3317800],
                                                [319700, 3317700]
                                            ]
                                        ],
                                        [
                                            [
                                                [319900, 3317900],
                                                [320000, 3317900],
                                                [320000, 3318000],
                                                [319900, 3318000],
                                                [319900, 3317900]
                                            ]
                                        ]
                                        ],
                                        "type": "MultiPolygon"
                                    }
                                    }
                                ],
                                "crs" :
                                {
                                    "type": "name",
                                    "properties": {
                                        "name": "EPSG:32636"
                                    }
                                }
                            },
                        }
                        "advanced":
                        {
                            "debug_with_roi": true
                        }
                    }

                Example of the "roi" parameter with a Shapefile

                .. code-block:: json

                    {
                        "inputs":
                        {
                            "roi" : "roi_vector_file.shp"
                        }
                    }

            .. tab:: Initial Elevation

                +----------------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+
                | Name                       | Description                                                         | Type                  | Default value        | Required |
                +============================+=====================================================================+=======================+======================+==========+
                | *initial_elevation*        | Low resolution DEM                                                  | See next section      | No                   | No       |
                +----------------------------+---------------------------------------------------------------------+-----------------------+----------------------+----------+

                **Initial elevation**

                The attribute contains all informations about initial elevation: dem path, geoid path and default altitudes.
                
                +-----------------------+----------------------------------------------------------------------------+--------+----------------------+----------------------+----------+
                | Name                  | Description                                                                | Type   | Available value      | Default value        | Required |
                +=======================+============================================================================+========+======================+======================+==========+
                | *dem*                 | Path to DEM file (one tile or VRT with concatenated tiles)                 | string |                      | None                 | No       |
                +-----------------------+----------------------------------------------------------------------------+--------+----------------------+----------------------+----------+
                | *geoid*               | Path to geoid file                                                         | string |                      | CARS internal geoid  | No       |
                +-----------------------+----------------------------------------------------------------------------+--------+----------------------+----------------------+----------+
                | *altitude_delta_min*  | Constant delta in altitude (meters) between *dem_median* and *dem_min*     | int    | should be > 0        | None                 | No       |
                +-----------------------+----------------------------------------------------------------------------+--------+----------------------+----------------------+----------+
                | *altitude_delta_max*  | Constant delta in altitude (meters) between *dem_max* and *dem_median*     | int    | should be > 0        | None                 | No       |
                +-----------------------+----------------------------------------------------------------------------+--------+----------------------+----------------------+----------+

                See section :ref:`download_srtm_tiles` to download 90-m SRTM DEM. If no DEM path is provided, an internal DEM is generated with sparse matches. Moreover, when there is no DEM data available, a default height above ellipsoid of 0 is used (no coverage for some points or pixels with no_data in the DEM tiles).
                
                If no geoid is provided, the default cars geoid is used (egm96).

                If no altitude delta is provided, the `dem_min` and `dem_max` generated with sparse matches will be used.
                
                The altitude deltas are used following this formula:

                .. code-block:: python

                    dem_min = initial_elevation - altitude_delta_min
                    dem_max = initial_elevation + altitude_delta_max

                .. warning::  DEM path is mandatory for the use of the altitude deltas.


                Initial elevation can be provided as a dictionary with a field for each parameter, for example:


                .. code-block:: json

                    {
                        "inputs": {
                            "initial_elevation": {
                                "dem": "/path/to/srtm.tif",
                                "geoid": "/path/to/geoid.tif",
                                "altitude_delta_min": 10,
                                "altitude_delta_max": 40
                            }
                        }
                    }

                Alternatively, it can be set as a string corresponding to the DEM path, in which case default values for the geoid and the default altitude are used.

                .. code-block:: json

                    {
                    "inputs": {
                            "initial_elevation": "/path/to/srtm.tif"
                        }
                    }

                Note that the `geoid` parameter in `initial_elevation` is not the geoid used for output products generated after the triangulation step
                (see output parameters).

                Elevation management is tightly linked to the geometry plugin used. See :ref:`plugins` section for details
		
    .. tab:: Orchestrator

        CARS can distribute the computations chunks by using either dask (local or distributed cluster) or multiprocessing libraries.
        The distributed cluster require centralized files storage and uses PBS scheduler.

        The ``orchestrator`` key is optional and allows to define orchestrator configuration that controls the distribution:

        +------------------+----------------------------------------------------------------------------------------------------------+-----------------------------------------+-----------------+----------+
        | Name             | Description                                                                                              | Type                                    | Default value   | Required |
        +==================+==========================================================================================================+=========================================+=================+==========+
        | *mode*           | Parallelization mode "local_dask", "pbs_dask", "slurm_dask", "multiprocessing", "auto" or "sequential"   | string                                  | "auto"          | Yes      |
        +------------------+----------------------------------------------------------------------------------------------------------+-----------------------------------------+-----------------+----------+
        | *task_timeout*   | Time (seconds) betweend two tasks before closing cluster and restarting tasks                            | int                                     | 600             | No       |
        +------------------+----------------------------------------------------------------------------------------------------------+-----------------------------------------+-----------------+----------+
        | *profiling*      | Configuration for CARS profiling mode                                                                    | dict                                    |                 | No       |
        +------------------+----------------------------------------------------------------------------------------------------------+-----------------------------------------+-----------------+----------+

        .. note::
            `sequential` orchestrator purposes are mostly for studies, debug and notebooks. If you want to use it with large data, consider using a ROI and Epipolar A Priori. Only tiles needed for the specified ROI will be computed. If Epipolar A priori is not specified, Epipolar Resampling and Sparse Matching will be performed on the whole image, no matter what ROI field is filled with.

        .. note::
            `auto` mode is a shortcut for *multiprocessing* orchestrator with parameters *nb_workers* and *max_ram_per_worker* set according to machine ressources and other parameters set to default value.
            This mode does not allow additional parameters.

            If CARS is launched on HPC cluster, this mode is not recommended because parameters would be set according to the full node resources.
            In this case, use multiprocessing mode and fill the parameters *nb_workers* and *max_ram_per_worker* according to the resources you requested.
    

        Depending on the used orchestrator mode, the following parameters can be added in the configuration:

        **Mode local_dask, pbs_dask:**

        +---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
        | Name                | Description                                                      | Type                                    | Default value | Required |
        +=====================+==================================================================+=========================================+===============+==========+
        | *nb_workers*        | Number of workers                                                | int, should be > 0                      | 2             | No       |
        +---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *max_ram_per_worker*| Maximum ram per worker                                           | int or float, should be > 0             | 2000          | No       |
        +---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *walltime*          | Walltime for one worker                                          | string, Should be formatted as HH:MM:SS | 00:59:00      | No       |
        +---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *use_memory_logger* | Usage of dask memory logger                                      | bool, True if use memory logger         | False         | No       |
        +---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *activate_dashboard*| Usage of dask dashboard                                          | bool, True if use dashboard             | False         | No       |
        +---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *python*            | Python path to binary to use in workers (not used in local dask) | str                                     | Null          | No       |
        +---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+


        **Mode slurm_dask:**

        +---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
        | Name                | Description                                                      | Type                                    | Default value | Required |
        +=====================+==================================================================+=========================================+===============+==========+
        | *account*           | SLURM account                                                    | str                                     |               | Yes      |
        +---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *nb_workers*        | Number of workers                                                | int, should be > 0                      | 2             | No       |
        +---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *max_ram_per_worker*| Maximum ram per worker                                           | int or float, should be > 0             | 2000          | No       |
        +---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *walltime*          | Walltime for one worker                                          | string, Should be formatted as HH:MM:SS | 00:59:00      | No       |
        +---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *use_memory_logger* | Usage of dask memory logger                                      | bool, True if use memory logger         | False         | No       |
        +---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *activate_dashboard*| Usage of dask dashboard                                          | bool, True if use dashboard             | False         | No       |
        +---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *python*            | Python path to binary to use in workers (not used in local dask) | str                                     | Null          | No       |
        +---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+
        | *qos*               | Quality of Service parameter (qos list separated by comma)       | str                                     | Null          | No       |
        +---------------------+------------------------------------------------------------------+-----------------------------------------+---------------+----------+


        **Mode multiprocessing:**

        +-----------------------+-----------------------------------------------------------+------------------------------------------+---------------+----------+
        | Name                  | Description                                               | Type                                     | Default value | Required |
        +=======================+===========================================================+==========================================+===============+==========+
        | *nb_workers*          | Number of workers                                         | int, should be > 0                       | 2             | No       |
        +-----------------------+-----------------------------------------------------------+------------------------------------------+---------------+----------+
        | *max_ram_per_worker*  | Maximum ram per worker                                    | int or float, should be > 0              | 2000          | No       |
        +-----------------------+-----------------------------------------------------------+------------------------------------------+---------------+----------+
        | *max_tasks_per_worker*| Number of tasks a worker can complete before refresh      | int, should be > 0                       | 10            | No       |
        +-----------------------+-----------------------------------------------------------+------------------------------------------+---------------+----------+
        | *dump_to_disk*        | Dump temporary files to disk                              | bool                                     | True          | No       |
        +-----------------------+-----------------------------------------------------------+------------------------------------------+---------------+----------+
        | *per_job_timeout*     | Timeout used for a job                                    | int or float                             | 600           | No       |
        +-----------------------+-----------------------------------------------------------+------------------------------------------+---------------+----------+
        | *factorize_tasks*     | Tasks sequentially dependent are run in one task          | bool                                     | True          | No       |
        +-----------------------+-----------------------------------------------------------+------------------------------------------+---------------+----------+
    
        .. note::

            **Factorisation**

            Two or more tasks are sequentially dependant if they can be run sequentially, independantly from any other task. 
            If it is the case, those tasks can be factorized, which means they can be run in a single task.
            
            Running several tasks in one task avoids doing useless dumps on disk between sequential tasks. It does not lose time 
            because tasks that are factorized could not be run in parallel, and it permits to save some time from the 
            creation of tasks and data transfer that are avoided.


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

        +---------------------+-----------------------------------------------------------+-----------------------------------------+----------------+----------+
        | Name                | Description                                               | Type                                    | Default value  | Required |
        +=====================+===========================================================+=========================================+================+==========+
        | *mode*              | type of profiling mode "cars_profiling, cprofile, memray" | string                                  | cars_profiling | No       |
        +---------------------+-----------------------------------------------------------+-----------------------------------------+----------------+----------+
        | *loop_testing*      | enable loop mode to execute each step multiple times      | bool                                    | False          | No       |
        +---------------------+-----------------------------------------------------------+-----------------------------------------+----------------+----------+

        - Please use make command 'profile-memory-report' to generate a memory profiling report from the memray outputs files (after the memray profiling execution).
        - Please disabled profiling to eval memory profiling at master orchestrator level and execute make command instead: 'profile-memory-all'.

        .. note::

            The logging system provides messages for all orchestration modes, both for the main process and the worker processes.
            The logging output file of the main process is located in the output directory.
            In the case of distributed orchestration, the worker's logging output file is located in the workers_log directory (the message format indicates thread ID and process ID).
            A summary of basic profiling is generated in output directory.

         
    .. tab:: Output


        +------------------+-------------------------------------------------------------+--------------------+----------------------+----------+
        | Name             | Description                                                 | Type               | Default value        | Required |
        +==================+=============================================================+====================+======================+==========+
        | *directory*      | Output folder where results are stored                      | string             | No                   | Yes      |
        +------------------+-------------------------------------------------------------+--------------------+----------------------+----------+
        | *product_level*  | Output requested products (dsm, point_cloud, depth_map)     | list or string     | "dsm"                | No       |
        +------------------+-------------------------------------------------------------+--------------------+----------------------+----------+
        | *resolution*     | Output DSM grid step (only for dsm product level)           | float              | 0.5                  | No       |
        +------------------+-------------------------------------------------------------+--------------------+----------------------+----------+
        | *auxiliary*      | Selection of additional files in products                   | dict               | See below            | No       |
        +------------------+-------------------------------------------------------------+--------------------+----------------------+----------+
        | *epsg*           | EPSG code                                                   | int, should be > 0 | None                 | No       |
        +------------------+-------------------------------------------------------------+--------------------+----------------------+----------+
        | *geoid*          | Output geoid                                                | bool or string     | False                | No       |
        +------------------+-------------------------------------------------------------+--------------------+----------------------+----------+
        | *save_by_pair*   | Save output point clouds by pair                            | bool               | False                | No       |
        +------------------+-------------------------------------------------------------+--------------------+----------------------+----------+

        .. code-block:: json

            {
                "output": {
                    "directory": "outresults",
                    "product_level": ["dsm", "depth_map"],
                    "geoid": true
                }
            }

        .. tabs::

            .. tab:: Output contents

                The output directory, defined on the configuration file contains at the end of the computation:

                * the required product levels (`depth_map`, `dsm` and/or `point_cloud`)
                * the dump directory (`dump_dir`) containing intermediate data for all applications
                * metadata json file (`metadata.json`) containing: used parameters, information and numerical results related to computation, step by step and pair by pair.
                * logs folder (`logs`) containing CARS log and profiling information


            .. tab:: Product level

                The `product_level` attribute defines which product should be produced by CARS. There are three available product type: `depth_map`, `point_cloud` and `dsm`.

                A single product can be requested by setting the parameter as string or several products can be requested by providing a list.

                .. tabs::

                    .. tab:: N pairs to 1 DSM

                        This is the default behavior of CARS : a single DSM will be generated from one or several pairs of images.

                        The smallest configuration can simply contain those inputs.

                        .. code-block:: json

                            {
                                "inputs": {
                                    "sensors" : {
                                        "one": {
                                            "image": "img1.tif",
                                            "geomodel": "img1.geom"
                                        },
                                        "two": {
                                            "image": "img2.tif",
                                            "geomodel": "img2.geom"

                                        },
                                        "three": {
                                            "image": "img3.tif",
                                            "geomodel": "img3.geom"
                                        }
                                    },
                                    "pairing": [["one", "two"],["one", "three"]]
                                }
                            }

                    .. tab:: N Depth Maps to 1 DSM

                        A single DSM will be generated from one or several depth_maps.

                        It is recommended to add the option ``"merging": true`` for this pipeline to improve performances.

                        .. code-block:: json

                            {
                                "inputs": {
                                    "depth_maps": {
                                        "my_name_for_this_depth_map": {
                                            "x" : "path_to_x.tif",
                                            "y" : "path_to_y.tif",
                                            "z" : "path_to_z.tif",
                                            "color" : "path_to_color.tif",
                                            "mask": "path_to_mask.tif",
                                            "classification": "path_to_classification.tif",
                                            "filling": "path_to_filling.tif",
                                            "confidence": {
                                                "confidence_name1": "path_to_confidence1.tif",
                                                "confidence_name2": "path_to_confidence2.tif",
                                            },
                                            "performance_map": "path_to_performance_map.tif",
                                            "epsg": "depth_map_epsg"
                                        }
                                    }
                                },
                                "advanced" {
                                    "merging": true
                                }
                            }

                    .. tab:: Sparse DSM

                        In CARS, sparse DSMs are computed during the process of creating depth maps from sensor images (specifically during the `dem_generation` application). This means they cannot be created from depth maps.
                        It also means the program should be stopped even before finishing the first part of the pipeline (sensor images to depth maps) in order not to run useless applications.

                        CARS provides an easy way of customizing the step at which the pipeline should be stopped. When the key ``product_level`` of ``output`` is empty, CARS will stop after the last application
                        whose ``save_intermediate_data`` key is set to True.

                        .. note::
                            If the sparse DSMs have already been created, they can then be re-entered in CARS through the ``terrain_a_priori`` parameter, saving computation time. File ``used_conf.json`` can be used directly by changing ``product_level`` and ``use_epipolar_a_priori`` parameters.
                            Very useful when trying to test multiple configurations later in the pipeline !

                        Applied to our current goal, this is the configuration needed to create sparse DSMs without useless applications running :

                        .. code-block:: json

                            {
                                "inputs": {
                                    "sensors" : {
                                        "one": {
                                            "image": "img1.tif",
                                            "geomodel": "img1.geom"
                                        },
                                        "two": {
                                            "image": "img2.tif",
                                            "geomodel": "img2.geom"

                                        }
                                    }
                                }
                                "applications": {
                                    "dem_generation": {
                                        "save_intermediate_data": true
                                    }
                                },
                                "output": {
                                    "product_level": []
                                }
                            }

                    .. tab:: N pairs to N Depth Maps

                        Depth maps are a way to represent point clouds as three images X Y and Z, each one representing the position of a pixel on its axis.
                        They are an official product of CARS, and can thus be created more easily than sparse DSMs.

                        The ``product_level`` key in ``output`` can contain any combination of the values `dsm`, `depth_map`, and `point_cloud`.

                        Depth maps (one for each sensor pair) will be saved if `depth_map` is present in ``product_level`` :

                        .. code-block:: json

                            {
                                "inputs": {
                                    "sensors" : {
                                        "one": {
                                            "image": "img1.tif",
                                            "geomodel": "img1.geom"
                                        },
                                        "two": {
                                            "image": "img2.tif",
                                            "geomodel": "img2.geom"

                                        }
                                    }
                                },
                                "output": {
                                    "product_level": ["depth_map"]
                                }
                            }

                    .. tab:: N pairs to N Point clouds

                        Just like depth maps, the point cloud is an official product of CARS. As such, all that's needed is to add `point_cloud` to ``product_level`` in order for it to be generated.

                        .. note::
                            A point cloud will be generated for each pair. If the ``merging`` parameter is activated, a single point cloud will be generated. However, this pipeline is not recommended because it uses a deprecated application.

                        .. code-block:: json

                             {
                                "inputs": {
                                    "sensors" : {
                                        "one": {
                                            "image": "img1.tif",
                                            "geomodel": "img1.geom"
                                        },
                                        "two": {
                                            "image": "img2.tif",
                                            "geomodel": "img2.geom"
                                        }
                                    }
                                }
                                "output": {
                                    "product_level": ["point_cloud"]
                                }
                            }

            .. tab:: Auxiliary data
                For `depth_map` and `dsm`, additional auxiliary files can be produced by setting the `auxiliary` dictionary attribute, it contains the following attributes:

                +-----------------------+-------------------------------------------------------------+--------+----------------+-----------+
                | Name                  | Description                                                 | Type   | Default value  | Required  |
                +=======================+=============================================================+========+================+===========+
                | *color*               | Save output color (dsm or depth_map)                        | bool   | True           | No        |
                +-----------------------+-------------------------------------------------------------+--------+----------------+-----------+
                | *mask*                | Save output mask (dsm or depth map)                         | bool   | False          | No        |
                +-----------------------+-------------------------------------------------------------+--------+----------------+-----------+
                | *classification*      | Save output classification (dsm or depth_map)               | bool   | False          | No        |
                +-----------------------+-------------------------------------------------------------+--------+----------------+-----------+
                | *performance_map*     | Save output performance map (dsm or depth_map)              | bool   | False          | No        |
                +-----------------------+-------------------------------------------------------------+--------+----------------+-----------+
                | *contributing_pair*   | Save output contributing pair (dsm)                         | bool   | False          | No        |
                +-----------------------+-------------------------------------------------------------+--------+----------------+-----------+
                | *filling*             | Save output filling (dsm or depth_map)                      | bool   | False          | No        |
                +-----------------------+-------------------------------------------------------------+--------+----------------+-----------+

                .. code-block:: json

                    {
                        "output": {
                            "directory": "outresults",
                            "product_level": ["dsm", "depth_map"],
                            "auxiliary": {"mask": true, "classification": true}
                        }
                    }

                Note that not all rasters associated to the DSM that CARS can produce are available in the output product auxiliary data. For example, confidence intervals are not part of the output product but can be found in the rasterization `dump_dir` if `generate_confidence_intervals` is activated in the `dense_matching` application (to compute the confidence) and `save_intermediate_data` is activated in the `rasterization` application configuration (to write it on disk).

            .. tab:: Geoid

                This parameter refers to the vertical reference of the output product, used as an altitude offset during triangulation.
                It can be set as a string to provide the path to a geoid file on disk, or as a boolean: if set to `True` CARS default geoid is used,
                if set to `False` no vertical offset is applied (ellipsoid reference).


            .. tab:: DSM output

                If product type `dsm` is selected, a directory named `dsm` will be created with the DSM and every auxiliary product selected. The file `dsm/index.json` shows the path of every generated file. For example :

                .. code-block:: json

                    {
                        "dsm": "dsm.tif",
                        "color": "color.tif",
                        "mask": "mask.tif",
                        "classification": "classification.tif",
                        "performance_map": "performance_map.tif",
                        "contributing_pair": "contributing_pair.tif",
                        "filling": "filling.tif"
                    }

            .. tab:: Depth map output

                If product type `depth_map` is selected, a directory named `depth_map` will be created with a subfolder for every pair. The file `depth_map/index.json` shows the path of every generated file. For example :

                .. code-block:: json

                    {
                        "one_two": {
                            "x": "one_two/X.tif",
                            "y": "one_two/Y.tif",
                            "z": "one_two/Z.tif",
                            "color": "one_two/color.tif",
                            "mask": "one_two/mask.tif",
                            "classification": "one_two/classification.tif",
                            "performance_map": "one_two/performance_map.tif",
                            "filling": "one_two/filling.tif",
                            "epsg": 4326
                        },
                        "one_three": {
                            "x": "one_three/X.tif",
                            "y": "one_three/Y.tif",
                            "z": "one_three/Z.tif",
                            "color": "one_three/color.tif",
                            "mask": "one_two/mask.tif",
                            "classification": "one_two/classification.tif",
                            "performance_map": "one_two/performance_map.tif",
                            "filling": "one_two/filling.tif",
                            "epsg": 4326
                        }
                    }

            .. tab:: Point cloud output

                If product type `point_cloud` is selected, a directory named `point_cloud` will be created with a subfolder for every pair.

                The point cloud output product consists of a collection of laz files, each containing a tile of the point cloud.

                The point cloud found in the product the highest level point cloud produced by CARS. For exemple, if outlier removal and point cloud denoising are deactivated, the point cloud will correspond to the output of triangulation. If only the first application of outlier removal is activated, this will be the output point cloud.

                The file `point_cloud/index.json` shows the path of every generated file. For example :

                .. code-block:: json

                    {
                        "one_two": {
                            "0_0": "one_two/0_0.laz",
                            "0_1": "one_two/0_1.laz"
                        },
                        "one_three": {
                            "0_0": "one_three/0_0.laz",
                            "0_1": "one_three/0_1.laz"
                        }
                    }


