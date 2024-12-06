.. _configuration:

Configuration
=============

This section describes CARS main configuration structure through a `json <http://www.json.org/json-fr.html>`_ configuration file.

The structure follows this organization:

.. code-block:: json

    {
        "inputs": {},
        "orchestrator": {},
        "applications": {},
        "output": {},
        "geometry_plugin": "geometry_plugin_to_use",
        "advanced": {}
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

            {
                "pipeline": "default"
            }

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

    .. tab:: Geometry plugin

        This section describes configuration of the geometry plugins for CARS, please refer to :ref:`plugins` section for details on plugins installation.

        +-------------------+-----------------------+--------+-------------------------+---------------------------------------+----------+
        | Name              | Description           | Type   | Default value           | Available values                      | Required |
        +===================+=======================+========+=========================+=======================================+==========+
        | *geometry_plugin* | The plugin to use     | str    | "SharelocGeometry"      | "SharelocGeometry"                    | False    |
        +-------------------+-----------------------+--------+-------------------------+---------------------------------------+----------+

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
              "geometry_plugin": "SharelocGeometry",
              "output": {
                "directory": "outresults"
              }
            }

        The particularities in the configuration file are:

        * **geomodel.model_type**: Depending on the nature of the geometric models indicated above, this field as to be defined as `RPC` or `GRID`. By default, "RPC".
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
                | method                               | Method for sparse matching                                                                     | string      | "sift"                 | "sift"        | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | disparity_margin                     | Add a margin to min and max disparity as percent of the disparity range.                       | float       |                        | 0.02          | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | elevation_delta_lower_bound          | Expected lower bound for elevation delta with respect to input low resolution dem in meters    | int, float  |                        | -9000         | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | elevation_delta_upper_bound          | Expected upper bound for elevation delta with respect to input low resolution dem in meters    | int, float  |                        | 9000          | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | epipolar_error_upper_bound           | Expected upper bound for epipolar error in pixels                                              | float       | should be > 0          | 10.0          | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | epipolar_error_maximum_bias          | Maximum bias for epipolar error in pixels                                                      | float       | should be >= 0         | 0.0           | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | disparity_outliers_rejection_percent | Percentage of outliers to reject                                                               | float       | between 0 and 1        | 0.1           | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | minimum_nb_matches                   | Minimum number of matches that must be computed to continue pipeline                           | int         | should be > 0          | 100           | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | sift_matching_threshold              | Threshold for the ratio to nearest second match                                                | float       | should be > 0          | 0.7           | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | sift_n_octave                        | The number of octaves of the Difference of Gaussians scale space                               | int         | should be > 0          | 8             | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | sift_n_scale_per_octave              | The numbers of levels per octave of the Difference of Gaussians scale space                    | int         | should be > 0          | 3             | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | sift_peak_threshold                  | Constrast threshold to discard a match (at None it will be set according to image type)        | float       | should be > 0          | 4.0           | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | sift_edge_threshold                  | Distance to image edge threshold to discard a match                                            | float       |                        | 10.0          | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | sift_magnification                   | The descriptor magnification factor                                                            | float       | should be > 0          | 7.0           | No       |
                +--------------------------------------+------------------------------------------------------------------------------------------------+-------------+------------------------+---------------+----------+
                | sift_window_size                     | smaller values let the center of the descriptor count more                                     | int         | should be > 0          | 2             | No       |
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


                For more information about these parameters, please refer to the `VLFEAT SIFT documentation <https://www.vlfeat.org/api/sift.html>`_.

                **Example**

                .. code-block:: json

                    "applications": {
                        "sparse_matching": {
                            "method": "sift",
                            "disparity_margin": 0.01
                        }
                    },

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

    .. tab:: Advanced parameters


        +----------------------------+-------------------------------------------------------------------------+-----------------------+----------------------+----------+
        | Name                       | Description                                                             | Type                  | Default value        | Required |
        +============================+=========================================================================+=======================+======================+==========+
        | *save_intermediate data*   | Save intermediate data for all applications                             | bool                  | False                | Yes      |
        +----------------------------+-------------------------------------------------------------------------+-----------------------+----------------------+----------+
        | *use_epipolar_a_priori*    | Active epipolar a priori                                                | bool                  | False                | Yes      |
        +----------------------------+-------------------------------------------------------------------------+-----------------------+----------------------+----------+
        | *epipolar_a_priori*        | Provide epipolar a priori information (see section below)               | dict                  |                      | No       |
        +----------------------------+-------------------------------------------------------------------------+-----------------------+----------------------+----------+
        | *terrain_a_priori*         | Provide terrain a priori information (see section below)                | dict                  |                      | No       |
        +----------------------------+-------------------------------------------------------------------------+-----------------------+----------------------+----------+
        | *debug_with_roi*           | Use input ROI with the tiling of the entire image (see Inputs section)  | bool                  | False                | No       |
        +----------------------------+-------------------------------------------------------------------------+-----------------------+----------------------+----------+
        | *merging*                  | Merge point clouds before rasterization (soon to be deprecated)         | bool                  | False                | No       |
        +----------------------------+-------------------------------------------------------------------------+-----------------------+----------------------+----------+


        **Save intermediate data**

        The `save_intermediate_data` flag can be used to activate and deactivate the saving of the possible output of applications.

        It is set in the `advanced` category and can be overloaded in each application separately. It default to false, meaning that no intermediate product in saved). Intermediate data are saved in the `dump_dir` folder found in CARS output directory, with a subfolder corresponding to each application.

        For exemple setting `save_intermediate_data` to `true` in `advanced` and to `false` in `application/point_cloud_rasterization` will activate product saving in all applications excepting `point_cloud_rasterization`. Conversely, setting it to `false` in `advanced` and to `true` in `application/point_cloud_rasterization`  will only save rasterization outputs.

        Intermediate data refers to all files that are not part of an output product. Files that compose an output product will not be found in the application dump directory. For exemple if `dsm` is requested as output product, the `dsm.tif` files and all activated dsm auxiliary files will not be found in `rasterization` dump directory. This directory will still contain the files generated by the `rasterization` application that are not part of the `dsm` product.

        .. code-block:: json

              "advanced": {
                  "save_intermediate_data": true
                  }
              }

        **Epipolar a priori**

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

        .. code-block:: json

              "advanced": {
                  "save_intermediate_data": true
                  }
              }

        **Terrain a priori**

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

        **Output contents**

        The output directory, defined on the configuration file contains at the end of the computation:

        * the required product levels (`depth_map`, `dsm` and/or `point_cloud`)
        * the dump directory (`dump_dir`) containing intermediate data for all applications
        * metadata json file (`metadata.json`) containing: used parameters, information and numerical results related to computation, step by step and pair by pair.
        * logs folder (`logs`) containing CARS log and profiling information


        **Output products**

        The `product_level` attribute defines which product should be produced by CARS. There are three available product type: `depth_map`, `point_cloud` and `dsm`.

        A single product can be requested by setting the parameter as string or several products can be requested by providing a list.

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
                    "product_level": "dsm",
                    "auxiliary": {"mask": true, "classification": true}
                }
            }

        Note that not all rasters associated to the DSM that CARS can produce are available in the output product auxiliary data. For example, confidence intervals are not part of the output product but can be found in the rasterization `dump_dir` if `generate_confidence_intervals` is activated in the `dense_matching` application (to compute the confidence) and `save_intermediate_data` is activated in the `rasterization` application configuration (to write it on disk).

        **Geoid**

        This parameter refers to the vertical reference of the output product, used as an altitude offset during triangulation.
        It can be set as a string to provide the path to a geoid file on disk, or as a boolean: if set to `True` CARS default geoid is used,
        if set to `False` no vertical offset is applied (ellipsoid reference).


        **DSM output**

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

        **Depth map output**

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

        **Point cloud output**

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

