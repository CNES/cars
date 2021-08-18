.. _user_manual_compute_cli:



.. _compute_dsm_cli:

Compute DSM pipeline CLI
========================

Command Description
-------------------

.. code-block:: console

        usage: cars compute_dsm [-h] -i [INJSONS [INJSONS ...]] -o OUTDIR
                                       [--sigma SIGMA] [--dsm_radius DSM_RADIUS]
                                       [--resolution RESOLUTION] [--epsg EPSG]
                                       [--roi_bbox ROI_BBOX ROI_BBOX ROI_BBOX ROI_BBOX | --roi_file ROI_FILE]
                                       [--dsm_no_data DSM_NO_DATA]
                                       [--color_no_data COLOR_NO_DATA]
                                       [--corr_config CORR_CONFIG]
                                       [--min_elevation_offset MIN_ELEVATION_OFFSET]
                                       [--max_elevation_offset MAX_ELEVATION_OFFSET]
                                       [--output_stats] [--use_geoid_as_alt_ref]
                                       [--use_sec_disp] [--snap_to_left_image]
                                       [--align_with_lowres_dem]
                                       [--disable_cloud_small_components_filter]
                                       [--disable_cloud_statistical_outliers_filter]
                                       [--mode {pbs_dask,local_dask,mp}]
                                       [--nb_workers NB_WORKERS] [--walltime WALLTIME]

        optional arguments:
        -h, --help            show this help message and exit
        --sigma SIGMA         Sigma for rasterization in fraction of pixels (default: None, should be >= 0)
        --dsm_radius DSM_RADIUS
                              Radius for rasterization in pixels (default: 1, should be >= 0)
        --resolution RESOLUTION
                              Digital Surface Model resolution (default: 0.5, should be > 0)
        --epsg EPSG           EPSG code (default: None, should be > 0)
        --roi_bbox ROI_BBOX ROI_BBOX ROI_BBOX ROI_BBOX
                              DSM ROI in final projection [xmin ymin xmax ymax] (it has to be in final projection)
        --roi_file ROI_FILE   DSM ROI file (vector file or image which footprint will be taken as ROI).
        --dsm_no_data DSM_NO_DATA
                              No data value to use in the final DSM file (default: -32768)
        --color_no_data COLOR_NO_DATA
                              No data value to use in the final color image (default: 0)
        --msk_no_data MSK_NO_DATA
                              No data value to use in the final mask image (default: 65535)
        --corr_config CORR_CONFIG
                              Correlator config (json file)
        --min_elevation_offset MIN_ELEVATION_OFFSET
                              Override minimum disparity from prepare step with this offset in meters
        --max_elevation_offset MAX_ELEVATION_OFFSET
                              Override maximum disparity from prepare step with this offset in meters
        --output_stats        Outputs dsm as a netCDF file embedding quality statistics.
        --use_geoid_as_alt_ref
                              Use geoid grid as altimetric reference.
        --use_sec_disp        Use the points cloudGenerated from the secondary disparity map.
        --snap_to_left_image  This mode can be used if all pairs share the same left image. It will then modify lines of sights of secondary images so that they all cross those of the reference image.
        --align_with_lowres_dem
                              If this mode is used, during triangulation, points will be corrected using the estimated correction from the prepare step in order to align 3D points with the low resolution initial
                              DEM.
        --disable_cloud_small_components_filter
                              This mode deactivates the points cloud filtering of small components.
        --disable_cloud_statistical_outliers_filter
                              This mode deactivates the points cloud filtering of statistical outliers.
        --mode {pbs_dask,local_dask,mp}
                              Parallelization mode (default: local_dask)
        --nb_workers NB_WORKERS
                              Number of workers (default: 2, should be > 0)
        --walltime WALLTIME   Walltime for one worker (default: 00:59:00). Should be formatted as HH:MM:SS)
        --loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                              Logger level (default: INFO. Should be one of (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        mandatory arguments:
          -i [INJSONS [INJSONS ...]], --injsons [INJSONS [INJSONS ...]]
                                Input json files
          -o OUTDIR, --outdir OUTDIR
                                Output directory


Command line usage:
-------------------

.. code-block:: console

    $ cars compute_dsm -i content.json content2.json ... -o outdir

This program takes as input a json file or a list of N json files in the case of a N images pairs processing. This corresponds to the content.json files generated at the prepare step (cf. above).
Its output is the path to the folder which will contain the results of the stereo, that is to say the ``dsm.tif`` (regular grid of altitudes) and the ``clr.tif`` (corresponding color) files.

Input optional parameters
-------------------------

Some optional parameters:

* ``sigma``: standard deviation value which controls the influence radius of each point of the cloud during the rasterization
* ``dsm_radius``: number of pixel rings to take into account in order to define the altitude of the current pixel
* ``resolution``: altitude grid step (dsm)
* ``epsg``: epsg code used for the cloud projection. If not set by the user, the more appropriate UTM zone will be retrieved automatically
* ``roi_bbox``: :term:`DSM` ROI in final projection [xmin ymin xmax ymax].

    * example with a quadruplet: ``cars compute_dsm content.json outdir/ --roi_bbox 0.1 0.2 0.3 0.4``
* ``roi_file`` : :term:`DSM` ROI file (vector file or image which footprint will be taken as ROI). The conversion to the final geometry ROI bounding box will be performed automatically. Mutually exclusive with ``roi_bbox`` option.
* ``dsm_no_data``: no data value of the final dsm
* ``color_no_data``: no data value of the final color ortho-image
* ``msk_no_data``: no data value to use in the final mask produced (if configured)
* ``corr_config``: correlator configuration file (for pandora)
* ``min_elevation_offset``: minimum offset in meter to use for the correlation. This parameter is converted in minimum of disparity using the disp_to_alt_ratio computed in the prepare step.
* ``max_elevation_offset``: maximum offset in meter to use for the correlation. This parameter is converted in maximum of disparity using the disp_to_alt_ratio computed in the prepare step.
* ``use_geoid_as_alt_ref``: controls the altimetric reference used to compute altitudes. If activated, the function uses the geoid file defined by the ```OTB_GEOID_FILE``` environment variable.
* ``use_sec_disp`` : enables to use the secondary disparity map to densify the 3D points cloud.
* ``snap_to_left_image`` : each 3D point is snapped to line of sight from left reference image (instead of using mid-point). This increases the coherence between several pairs if left image is the same image for all pairs.
* ``align_with_lowres_dem``: During prepare step, a cubic splines correction is computed so as to align :term:`DSM` from a pair with the initial low resolution DEM. If this mode is used, the correction estimated for each pair is applied. This will increases coherency between pairs and with the initial low resolution DEM.
* ``disable_cloud_small_components_filter``: Deactivate the filtering of small 3D points groups. The filtered groups are composed of less than 50 points, the distance between two "linked" points is less than 3.
* ``disable_cloud_statistical_outliers_filter``: Deactivate the statistical filtering of the 3D points. For this filter the examined statistic is the mean distance of each point to its 50 nearest neighbors. The filtered points have a mean distance superior than this statistic's mean + 5 * this statistic's standard deviation.

Cluster parameters
------------------

During its execution, this program creates a distributed dask cluster (except if the ``mode`` is ``mp``).

The following parameters can be used :

* ``mode``: parallelisation mode (``pbs_dask``, ``local_dask`` or ``mp`` for multiprocessing)
* ``nb_workers``: dask cluster or multiprocessing workers number
* ``walltime``: maximum time of execution

.. note::

  Using INFO loglevel parameter, a dask dashboard URL is displayed in the logs to follow the dask tasks execution in real time.


To know the number of used cores, the program is using the ``OMP_NUM_THREADS`` environment variable.
In CARS code, the tile size is estimated from the value of the ``OTB_MAX_RAM_HINT`` variable (expressed in MB) times the memory amount reserved for a node, it is to say ``OMP_NUM_THREADS x 5 Gb``.
For a production at full image scale (or using several images), it is recommended that ``OTB_MAX_RAM_HINT`` is set to a value high enough to fill the allocated resources. For example, for ``OMP_NUM_THREADS=8``, the allocated memory for a node is set to 20Gb, thus the ``OTB_MAX_RAM_HINT`` can be set to 10 000.
A low value of ``OTB_MAX_RAM_HINT`` leads to a higher number of generated tiles and an under-consumption of the allocated resources.

Other environment variables can impact the dask execution on the cluster:

* ``CARS_NB_WORKERS_PER_PBS_JOB``: defines the number of workers that are started for each PBS job (set to 2 by default)
* ``CARS_PBS_QUEUE``: enables to turn to another queue than the standard one (dev for example)
* ``OPJ_NUM_THREADS``, ``NUMBA_NUM_THREADS`` and ``GDAL_NUM_THREADS`` are exported on each job (all set by default to the same value as ``OMP_NUM_THREADS``, it is to say 4)

The nodes on which the computations are performed should be able to handle the opening of several files at once. In the other case, some "Too many open files" errors can happen. It is then recommended to launch the command again on nodes which have a higher opened files limit.

Loglevel parameter
------------------
The ``loglevel`` option allows to parameter the loglevel. By default, the WARNING loglevel gives few information: only criticals, errors and warnings execution messages.

.. note::

	Use ``cars compute_dsm -i input.json -o outdir --loglevel INFO`` to get many detailed information about each CARS steps.


Output contents
---------------

The output folder contains a content.json file, the computed dsm, the color ortho-image (if the ``color1`` field is not set in the input configuration file then the ``img1`` is used) and, if dask is used, the dask configuration.

.. code-block:: console

    $ ls
    yy-MM-dd_HHhmmm_compute_dsm.log  clr.tif  content.json  dask_config_compute_dsm.yaml  dask_log


If the ``--output_stats`` is activated, the output directory will contain tiff images corresponding to different statistics computed during the rasterization.

.. code-block:: console

    $ ls
    yy-MM-dd_HHhmmm_compute_dsm.log  clr.tif  content.json  dask_config_compute_dsm.yaml  dask_log  dsm_mean.tif  dsm_n_pts.tif  dsm_pts_in_cell.tif  dsm_std.tif  dsm.tif

Those statistics are:

* The number of 3D points used to compute each cell (``dsm_n_pts.tif``)
* The elevations's mean of the 3D points used to compute each cell (``dsm_mean.tif``)
* The elevations's standard deviation of the 3D points used to compute each cell (``dsm_std.tif``)
* The number of 3D points strictly contained in each cell (``dsm_pts_in_cell.tif``)


Once the computation is done, the output folder also contains a ``content.json`` file describing the folder's content and reminding the complete history of the production.

.. code-block:: json

    {
      "input_configurations": [
        {
          "input_configuration": {
            "input": {
              "img1": "/tmp/cars/tests/data/input/phr_paca/left_image.tif",
              "mask1": "/tmp/cars/tests/data/input/phr_paca/left_multiclass_msk.tif",
              "mask1_classes": "/tmp/cars/tests/data/input/phr_paca/left_msk_classes.json",
              "nodata1": 0,
              "img2": "/tmp/cars/tests/data/input/phr_paca/right_image.tif",
              "mask2": "/tmp/cars/tests/data/input/phr_paca/right_multiclass_msk.tif",
              "mask2_classes": "/tmp/cars/tests/data/input/phr_paca/right_msk_classes.json",
              "nodata2": 0,
              "srtm_dir": "/tmp/cars/tests/data/input/phr_paca/srtm"
            },
            "preprocessing": {
              "version": "147_multi_classes_mask_doc//847e",
              "parameters": {
                "epi_step": 30,
                "disparity_margin": 0.25,
                "epipolar_error_upper_bound": 43.0,
                "epipolar_error_maximum_bias": 0.0,
                "elevation_delta_lower_bound": -20.0,
                "elevation_delta_upper_bound": 20.0,
                "mask_classes_usage_in_prepare": {
                  "mask1_ignored_by_sift_matching": [
                    1
                  ],
                  "mask2_ignored_by_sift_matching": [
                    1
                  ]
                }
              },
              "static_parameters": {
                "sift": {
                  "matching_threshold": 0.6,
                  "n_octave": 8,
                  "n_scale_per_octave": 3,
                  "dog_threshold": 20.0,
                  "edge_threshold": 5.0,
                  "magnification": 2.0,
                  "back_matching": true
                },
                "low_res_dsm": {
                  "low_res_dsm_resolution_in_degree": 0.000277777777778,
                  "lowres_dsm_min_sizex": 100,
                  "lowres_dsm_min_sizey": 100,
                  "low_res_dsm_ext": 3,
                  "low_res_dsm_order": 3
                },
                "disparity_range": {
                  "disparity_outliers_rejection_percent": 0.1
                }
              },
              "output": {
                "left_envelope": "/tmp/out_preproc/left_envelope.shp",
                "right_envelope": "/tmp/out_preproc/right_envelope.shp",
                "envelopes_intersection": "/tmp/out_preproc/envelopes_intersection.gpkg",
                "envelopes_intersection_bounding_box": [
                  7.292954644352718,
                  43.68961593954899,
                  7.295742924906745,
                  43.691746080922535
                ],
                "epipolar_size_x": 550,
                "epipolar_size_y": 550,
                "epipolar_origin_x": 0.0,
                "epipolar_origin_y": 0.0,
                "epipolar_spacing_x": 30.0,
                "epipolar_spacing_y": 30.0,
                "disp_to_alt_ratio": 1.342233116897663,
                "left_azimuth_angle": 324.2335255560172,
                "left_elevation_angle": 79.63809387446263,
                "right_azimuth_angle": 223.4124262214363,
                "right_elevation_angle": 73.44127819956262,
                "convergence_angle": 21.049281048130418,
                "raw_matches": "/tmp/out_preproc/raw_matches.npy",
                "left_epipolar_grid": "/tmp/out_preproc/left_epipolar_grid.tif",
                "right_epipolar_grid": "/tmp/out_preproc/right_epipolar_grid.tif",
                "right_epipolar_uncorrected_grid": "/tmp/out_preproc/right_epipolar_grid_uncorrected.tif",
                "minimum_disparity": -14.42170348554717,
                "maximum_disparity": 12.408438545673961,
                "matches": "/tmp/out_preproc/matches.npy",
                "lowres_dsm": "/tmp/out_preproc/lowres_dsm_from_matches.nc",
                "lowres_initial_dem": "/tmp/out_preproc/lowres_initial_dem.nc",
                "lowres_elevation_difference": "/tmp/out_preproc/lowres_elevation_diff.nc"
              }
            }
          },
          "mask_classes_usage_in_compute_dsm": {
            "mask1_ignored_by_correlation": [
              1
            ],
            "mask1_set_to_ref_alt": [
              1
            ],
            "mask2_ignored_by_correlation": [
              1
            ],
            "mask2_set_to_ref_alt": [
              1,
              150
            ]
          }
        }
      ],
      "stereo": {
        "version": "147_multi_classes_mask_doc//847e",
        "parameters": {
          "resolution": 0.5,
          "sigma": 0.3,
          "dsm_radius": 3,
          "epsg": 32631
        },
        "static_parameters": {
          "tiling_configuration": {
            "epipolar_tile_margin_in_percent": 20
          },
          "rasterization": {
            "grid_points_division_factor": null
          },
          "cloud_filtering": {
            "small_components": {
              "on_ground_margin": 10,
              "connection_distance": 3.0,
              "nb_points_threshold": 50,
              "clusters_distance_threshold": null,
              "removed_elt_mask": false,
              "mask_value": 255
            },
            "statistical_outliers": {
              "k": 50,
              "std_dev_factor": 5.0,
              "removed_elt_mask": false,
              "mask_value": 255
            }
          },
          "output": {
            "color_image_encoding": "uint16"
          },
          "loaders":{
            "geometry": "OTBGeometry"
          },
          "geoid_path": 'path/to/geoid'
        },
        "output": {
          "altimetric_reference": "ellipsoid",
          "epsg": 32631,
          "dsm": "dsm.tif",
          "dsm_no_data": -999.0,
          "color_no_data": 0.0,
          "color": "clr.tif",
          "msk": "/tmp/out_stereo/msk.tif",
          "dsm_mean": "dsm_mean.tif",
          "dsm_std": "dsm_std.tif",
          "dsm_n_pts": "dsm_n_pts.tif",
          "dsm_points_in_cell": "dsm_pts_in_cell.tif"
        }
      }
    }


.. _`GDAL`: https://gdal.org/
