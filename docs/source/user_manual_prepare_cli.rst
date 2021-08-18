.. _user_manual_prepare_cli:


Prepare pipeline CLI
====================

Command Description
-------------------

.. code-block:: console

        usage: cars prepare [-h] -i INJSON -o OUTDIR [--epi_step EPI_STEP]
                                   [--disparity_margin DISPARITY_MARGIN]
                                   [--epipolar_error_upper_bound EPIPOLAR_ERROR_UPPER_BOUND]
                                   [--epipolar_error_maximum_bias EPIPOLAR_ERROR_MAXIMUM_BIAS]
                                   [--elevation_delta_lower_bound ELEVATION_DELTA_LOWER_BOUND]
                                   [--elevation_delta_upper_bound ELEVATION_DELTA_UPPER_BOUND]
                                   [--mode {pbs_dask,local_dask}]
                                   [--nb_workers NB_WORKERS] [--walltime WALLTIME]
                                   [--check_inputs]

        optional arguments:
          -h, --help            show this help message and exit
          --epi_step EPI_STEP   Step of the deformation grid in nb. of pixels (default: 30, should be > 1)
          --disparity_margin DISPARITY_MARGIN
                                Add a margin to min and max disparity as percent of the disparity range (default: 0.02, should be in range [0,1])
          --epipolar_error_upper_bound EPIPOLAR_ERROR_UPPER_BOUND
                                Expected upper bound for epipolar error in pixels (default: 10, should be > 0)
          --epipolar_error_maximum_bias EPIPOLAR_ERROR_MAXIMUM_BIAS
                                Maximum bias for epipolar error in pixels (default: 0, should be >= 0)
          --elevation_delta_lower_bound ELEVATION_DELTA_LOWER_BOUND
                                Expected lower bound for elevation delta with respect to input low resolution DTM in meters (default: -1000)
          --elevation_delta_upper_bound ELEVATION_DELTA_UPPER_BOUND
                                Expected upper bound for elevation delta with respect to input low resolution DTM in meters (default: 1000)
          --mode {pbs_dask,local_dask}
                                Parallelization mode (default: local_dask)
          --nb_workers NB_WORKERS
                                Number of workers (default: 2, should be > 0)
          --walltime WALLTIME   Walltime for one worker (default: 00:59:00). Should be formatted as HH:MM:SS)
          --check_inputs        Check inputs consistency
          --loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                                Logger level (default: INFO. Should be one of (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        mandatory arguments:
          -i INJSON, --injson INJSON
                                Input json file
          -o OUTDIR, --outdir OUTDIR
                                Output directory


Command line usage
------------------

.. code-block:: console

    $ cars prepare -i preproc_input.json -o outdir


Input json file
---------------

The prepare input file (``preproc_input.json``) file is formatted as follows:

.. code-block:: json

    {
        "img1" : "/tmp/cars/tests/data/input/phr_paca/left_image.tif",
        "color1" : "/tmp/cars/tests/data/input/phr_paca/left_image.tif",
        "img2" : "/tmp/cars/tests/data/input/phr_paca/right_image.tif",
        "mask1" : "/tmp/cars/tests/data/input/phr_paca/left_multiclasses_msk.tif",
        "mask2" : "/tmp/cars/tests/data/input/phr_paca/right_multiclasses_msk.tif",
        "mask1_classes" : "/tmp/cars/tests/data/input/phr_paca/left_msk_classes.json",
        "mask2_classes" : "/tmp/cars/tests/data/input/phr_paca/right_msk_classes.json",
        "srtm_dir" : "/tmp/cars/tests/data/input/phr_paca/srtm",
        "default_alt": 0,
        "nodata1": 0,
        "nodata2": 0
    }



The mandatory fields of the input json file are:

* The ``img1`` and ``img2`` fields contain the paths to the images forming the pair.
* ``nodata1`` : no data value of the image 1.
* ``nodata2`` : no data value of the image 2.

The other optional fields of the input json file are:

* The ``srtm_dir`` field contains the path to the folder in which are located the srtm tiles covering the production.
* ``default_alt`` : this parameter allows to set the default height above ellipsoid when there is no DEM available, no coverage for some points or pixels with no_data in the DEM tiles (default value: 0).
* ``mask1`` : external mask of the image 1. This mask can be a "two-states" mask (convention: 0 is a valid pixel, other values indicate data to ignore) or a multi-classes mask in which case the ``mask1_classes`` shall be indicated in the configuration file.
* ``mask2`` : external mask of the image 2. This mask can be a "two-states" mask (convention: 0 is a valid pixel, other values indicate data to ignore) or a multi-classes mask in which case the ``mask2_classes`` shall be indicated in the configuration file.
* ``mask1_classes`` : json file indicated the ``mask1``'s classes usage (see next section for more details).
* ``mask2_classes`` : json file indicated the ``mask2``'s classes usage (see next section for more details).
* ``color1`` : image stackable to ``img1`` used to create an ortho-image corresponding to the produced :term:`DSM`. This image can be composed of XS bands in which case a PAN+XS fusion will be performed.


**Warning** : If the ``mask1`` (or ``mask2``) is a multi-classes one and no ``mask1_classes`` (or ``mask2_classes``) configuration file is indicated, all non-zeros values of the mask will be considered as unvalid data.

**Warning** : The value 255 is reserved for CARS internal use, thus no class can be represented by this value in the masks.


CARS mask multi-classes json file
---------------------------------

Multi-classes masks have a unified CARS json format enabling the use of several mask information into the API.
The classes can be used in different ways depending on the tag used in the json file defined below.

Json files are given in the ``mask1_classes`` and ``mask2_classes`` fields of the configuration files (see previous section).
These files indicate the masks's classes usage and are structured as follows :

.. code-block:: json

    {
        "ignored_by_correlation": [1, 2],
        "set_to_ref_alt": [1, 3, 4],
        "ignored_by_sift_matching": [2]
    }

Usage in the ``prepare`` step:

* The classes listed in ``ignored_by_sift_matching`` will be masked at the sparse matching step.

Usage in the ``compute_dsm`` step:

* The classes listed in ``ignored_by_correlation`` will be masked at the correlation step (pandora).
* The classes listed in ``set_to_ref_alt`` will be set to the reference altitude (srtm or scalar). To do so, these pixels's disparity will be set to 0.


Input optional parameters
-------------------------

Some optional parameters of the command line impact the matching:

* ``epi_step`` parameter :  step of the epipolar grid to compute (in pixels in epipolar geometry).
* ``disparity_margin`` parameter :  Add a margin to min and max disparity as percent of the disparity range.
* ``epipolar_error_upper_bound`` parameter: expected epipolar error upper bound (in pixels).
* ``epipolar_error_maximum_bias`` parameter: value added to the vertical margins for the sparse matching. If this parameter is different to zero then the shift produced by an potential bias on the geometrical models is compensated by taking into account the median shift computed from the img1 and img2 matches.
* ``elevation_delta_lower_bound`` parameter: expected lower bound of the altitude discrepancy with the input DEM (in meters).
* ``elevation_delta_upper_bound`` parameter: expected upper bound of the altitude discrepancy with the input DEM (in meters).

Cluster parameters
------------------

During its execution, this program creates a distributed dask cluster.

The following parameters can be used :

* ``mode``: parallelisation mode (``pbs_dask`` or  ``local_dask``)
* ``nb_workers``: dask cluster workers number
* ``walltime``: maximum time of execution

.. note::

  Using INFO loglevel parameter, a dask dashboard URL is displayed in the logs to follow the dask tasks execution in real time.

Check inputs parameter
----------------------

``cars prepare`` has also a ``--check_inputs`` option to improve input data consistency checking:

* ``img1`` and ``img2`` only have one band, are readable with the OTB and have a RPC model. It is also checked that the data seem to be in the sensor geometry (positive pixel size).
* ``mask1`` has the same size as ``img1`` and that ``mask2`` has the same size as ``img2``.
* the ground intersection zone between ``img1`` and ``img2`` is not empty.
* the srtm given in input covers the ground intersection zone of ``img1`` and ``img2``. For information purposes, if it is not equal to 100%, the coverage ratio of the dem with respect to the useful zone is given in the logs (INFO loglevel).

By default this option is **deactivated** because it can be potentially time-consuming.

Loglevel parameter
------------------
The ``loglevel`` option allows to parameter the loglevel. By default, the WARNING loglevel gives few information: only criticals, errors and warnings execution messages.

.. note::

	Use ``cars prepare -i input.json -o outdir --loglevel INFO`` to get many detailed information about each CARS steps.

Output contents
---------------

After its execution, the ``outdir`` folder contains the following elements:

.. code-block:: console

    ls outdir/
    yy-MM-dd_HHhmmm_prepare.log  dask_log                     left_envelope.prj  left_epipolar_grid.tif      lowres_initial_dem.nc  right_envelope.dbf  right_envelope.shx
    content.json                 envelopes_intersection.gpkg  left_envelope.shp  lowres_dsm_from_matches.nc  matches.npy            right_envelope.prj  right_epipolar_grid.tif
    dask_config_prepare.yaml     left_envelope.dbf            left_envelope.shx  lowres_elevation_diff.nc    raw_matches.npy        right_envelope.shp  right_epipolar_grid_uncorrected.tif


The ``content.json`` file lists the generated files and some numerical elements:

.. code-block:: json

    {
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
          },
          "loaders":{
            "geometry": "OTBGeometry"
          },
          "geoid_path": 'path/to/geoid'
        },
        "output": {
          "left_envelope": "left_envelope.shp",
          "right_envelope": "right_envelope.shp",
          "envelopes_intersection": "envelopes_intersection.gpkg",
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
          "raw_matches": "raw_matches.npy",
          "left_epipolar_grid": "left_epipolar_grid.tif",
          "right_epipolar_grid": "right_epipolar_grid.tif",
          "right_epipolar_uncorrected_grid": "right_epipolar_grid_uncorrected.tif",
          "minimum_disparity": -14.42170348554717,
          "maximum_disparity": 12.408438545673961,
          "matches": "matches.npy",
          "lowres_dsm": "lowres_dsm_from_matches.nc",
          "lowres_initial_dem": "lowres_initial_dem.nc",
          "lowres_elevation_difference": "lowres_elevation_diff.nc"
        }
      }
    }


The other files are:

* ``left_epipolar_grid.tif`` : left image epipolar grid
* ``right_epipolar_grid.tif`` : right image epipolar grid with correction
* ``left_envelope.shp`` : left image envelope
* ``right_envelope.shp`` : right image envelope
* ``envelopes_intersection.gpkg`` : intersection of the right and left images's envelopes
* ``ground_positions_grid.tif`` : image with the same geometry as the epipolar grid and for which each point has for value the ground position (lat/lon) of the corresponding point in the epipolar grid
* ``matches.npy`` : matches list after filtering
* ``raw_matches.npy`` : initial raw matches list
* ``lowres_dsm_from_matches.nc`` : low resolution :term:`DSM` computed from the matches
* ``lowres_elevation_diff.nc`` : difference between the low resolution :term:`DSM` computed from the matches and the initial DEM in input of the prepare step
* ``lowres_initial_dem.nc`` : initial DEM in input of the prepare step corresponding to the two images envelopes's intersection zone
* ``corrected_lowres_dsm_from_matches.nc`` :  Corrected low resolution :term:`DSM` from matches if low resolution :term:`DSM` is large enough (minimum size is 100x100)
* ``corrected_lowres_elevation_diff.nc`` : difference between the initial DEM in input of the prepare step  and the corrected low resolution :term:`DSM`. if low resolution :term:`DSM` is large enough (minimum size is 100x100)
* ``dask_config_prepare.yaml`` : the dask configuration used (only for ``local_dask`` and ``pbs_dask`` modes)
