#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of CARS
# (see https://github.com/CNES/cars).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import argparse
import warnings
import argcomplete

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def cars_cli_parser():
    # Create cars cli parser fril argparse
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="CARS: CNES Algorithms to Reconstruct Surface")

    # General arguments at first level
    parser.add_argument(
        "--loglevel", default="INFO",
       choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
       help="Logger level (default: INFO. Should be one of "
            "(DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    # Create subcommand parser for prepare and compute_dsm
    subparsers = parser.add_subparsers(dest="command")

    ### Prepare subcommand ###

    # Prepare subparser creation
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Preparation for compute_dsm "
             "producing stereo-rectification grid as well "
             "as an estimate of the disparity to explore.")

    ## Prepare arguments

    # Required
    prepare_parser.add_argument(
        "-i", "--injson", required=True, type=str,  help="Input json file")
    prepare_parser.add_argument(
        "-o", "--outdir", required=True, type=str,  help="Output directory")

    # Optionals
    prepare_parser.add_argument(
        "--epi_step", type=int, default=30,
        help="Step of the deformation grid in nb. of pixels "
             "(default: 30, should be > 1)")
    prepare_parser.add_argument(
        "--disparity_margin", type=float, default=0.02,
        help="Add a margin to min and max disparity as percent "
             "of the disparity range "
             "(default: 0.02, should be in range [0,1])")
    prepare_parser.add_argument(
        "--epipolar_error_upper_bound",
        type=float, default=10.,
        help="Expected upper bound for epipolar error in pixels "
             "(default: 10, should be > 0)")
    prepare_parser.add_argument(
        "--epipolar_error_maximum_bias",
        type=float, default=0.,
        help="Maximum bias for epipolar error in pixels " 
             "(default: 0, should be >= 0)")
    prepare_parser.add_argument(
        "--elevation_delta_lower_bound", type=float, default=-1000.,
        help="Expected lower bound for elevation delta "
             "with respect to input low resolution DTM in meters "
             "(default: -1000)")
    prepare_parser.add_argument(
        "--elevation_delta_upper_bound", type=float, default=1000.,
        help="Expected upper bound for elevation delta "
             "with respect to input low resolution DTM in meters "
             "(default: 1000)")
    prepare_parser.add_argument(
        "--mode", default="local_dask",
        choices=("pbs_dask", "local_dask"),
        help="Parallelization mode (default: local_dask)")
    prepare_parser.add_argument(
        "--nb_workers", type=int, default=8,
        help="Number of workers (default: 8, should be > 0)")
    prepare_parser.add_argument(
        "--walltime", default="00:59:00",
        help="Walltime for one worker (default: 00:59:00). "
             "Should be formatted as HH:MM:SS)")
    prepare_parser.add_argument(
        "--check_inputs",action="store_true",
        help="Check inputs consistency")

    ### Compute_dsm subcommand ###

    # Prepare subparser creation
    compute_dsm_parser = subparsers.add_parser(
        "compute_dsm",
        help="Tile-based, concurent resampling "
             "in epipolar geometry, disparity "
             "estimation, triangulation and rasterization")

    ## Compute_dsm arguments

    # Required
    compute_dsm_parser.add_argument(
        "-i","--injsons", required=True, help="Input json files", nargs='*')
    compute_dsm_parser.add_argument(
        "-o","--outdir", required=True, type=str, help="Output directory")

    # Optionals
    compute_dsm_parser.add_argument(
        "--sigma", type=float, default=None,
        help="Sigma for rasterization in fraction of pixels "
             "(default: None, should be >= 0)")
    compute_dsm_parser.add_argument(
        "--dsm_radius", type=int, default=1,
        help="Radius for rasterization in pixels "
             "(default: 1, should be >= 0)")
    compute_dsm_parser.add_argument(
        "--resolution", type=float, default=0.5,
        help="Digital Surface Model resolution "
             "(default: 0.5, should be > 0)")
    compute_dsm_parser.add_argument(
        "--epsg", type=int, default=None,
        help="EPSG code (default: None, should be > 0)")
    compute_dsm_parser.add_argument(
        "--roi", default=None, nargs='*',
        help="DSM ROI in final projection [xmin ymin xmax ymax] "
             "(it has to be in final projection) or DSM ROI file "
             "(vector file or image which "
             "footprint will be taken as ROI).")
    compute_dsm_parser.add_argument(
        "--dsm_no_data", type=int, default=-32768,
        help="No data value to use in the final DSM file (default: -32768)")
    compute_dsm_parser.add_argument(
        "--color_no_data", type=int, default=0,
        help="No data value to use in the final color image (default: 0)")
    compute_dsm_parser.add_argument(
        "--corr_config", default=None, type=str,
        help="Correlator config (json file)")
    compute_dsm_parser.add_argument(
        "--min_elevation_offset", type=float, default=None,
        help="Override minimum disparity from prepare step "
             "with this offset in meters")
    compute_dsm_parser.add_argument(
        "--max_elevation_offset", type=float, default=None,
        help="Override maximum disparity from prepare step "
             "with this offset in meters")
    compute_dsm_parser.add_argument(
        "--output_stats", action="store_true",
        help="Outputs dsm as a netCDF file embedding quality statistics.")
    compute_dsm_parser.add_argument(
        "--use_geoid_as_alt_ref", action="store_true", default=False,
        help="Use geoid grid as altimetric reference.")
    compute_dsm_parser.add_argument(
        "--use_sec_disp", action="store_true",
        help="Use the points cloud"
             "Generated from the secondary disparity map.")
    compute_dsm_parser.add_argument(
        "--snap_to_left_image", action='store_true', default=False,
        help="This mode can be used if all pairs share the same left image. "
             "It will then modify lines of sights of secondary images "
             "so that they all cross those of the reference image.")
    compute_dsm_parser.add_argument(
        "--align_with_lowres_dem", action='store_true', default=False,
        help="If this mode is used, during triangulation, "
             "points will be corrected using the estimated correction "
             "from the prepare step in order to align 3D points "
             "with the low resolution initial DEM.")
    compute_dsm_parser.add_argument(
        "--disable_cloud_small_components_filter", action="store_true",
        help="This mode deactivates the points cloud "
             "filtering of small components.")
    compute_dsm_parser.add_argument(
        "--disable_cloud_statistical_outliers_filter", action='store_true',
        help="This mode deactivates the points cloud "
             "filtering of statistical outliers.")
    compute_dsm_parser.add_argument(
        "--mode",default="local_dask",
        choices=("pbs_dask", "local_dask", "mp"),
        help="Parallelization mode (default: local_dask) ")
    compute_dsm_parser.add_argument(
        "--nb_workers", type=int, default=32, 
        help="Number of workers (default: 32, should be > 0)")
    compute_dsm_parser.add_argument(
        "--walltime", default="00:59:00",
        help="Walltime for one worker (default: 00:59:00). "
             "Should be formatted as HH:MM:SS)")

    # autocomplete
    argcomplete.autocomplete(parser)

    return parser


def parse_roi_argument(roi_args, stop_now):
    """
    Parse ROI argument

    :param roi_args: ROI argument
    :type region: str or array of four numbers
    :param stop_now: Argument check
    :type stop_now: Boolean
    :return: ROI (Bounds, EPSG code)
    :rtype: Tuple with array of 4 floats and int
    """
    import logging
    import rasterio
    from cars import utils

    roi = None
    if roi_args is not None:
        if len(roi_args) == 1:
            # in case the input is a string
            if isinstance(roi_args[0], str):
                roi_file = roi_args[0]
                name, extension = os.path.splitext(roi_file)

                # test file existence
                if not os.path.exists(roi_file):
                    logging.warning('{} does not exist'.format(roi_file))
                    stop_now = True

                # if it is a vector file
                if extension in ['.gpkg', '.shp', '.kml']:
                    try:
                        roi_poly, roi_epsg = utils.read_vector(roi_file)
                        roi = (roi_poly.bounds, roi_epsg)
                    except BaseException:
                        logging.critical(
                            'Impossible to read {} file'.format(roi_args))
                        stop_now = True

                # if not, it is an image
                elif utils.rasterio_can_open(roi_file):
                    data = rasterio.open(roi_file)
                    xmin = min(data.bounds.left, data.bounds.right)
                    ymin = min(data.bounds.bottom, data.bounds.top)
                    xmax = max(data.bounds.left, data.bounds.right)
                    ymax = max(data.bounds.bottom, data.bounds.top)

                    try:
                        roi_epsg = data.crs.to_epsg()
                        roi = ([xmin, ymin, xmax, ymax], roi_epsg)
                    except AttributeError as e:
                        logging.critical(
                            'Impossible to read the ROI image epsg code: {}'.format(e))
                        stop_now = True

                else:
                    logging.critical(
                        '{} has an unsupported file format'.format(roi_args))
                    stop_now = True

        elif len(roi_args) == 4:
            # in case the input has a [xmin, ymin, xmax, ymax] ROI
            try:
                roi = ([float(elt) for elt in roi_args], None)
            except BaseException:
                logging.critical('Cannot parse {} argument'. format(roi_args))
                stop_now = True
            logging.warning('Input ROI shall be in final projection')
        else:
            logging.critical('--roi is not set properly')
            stop_now = True
    return roi, stop_now


def main_cli(args, parser, check_inputs=False):
    """
    Main for command line management

    :param check_inputs: activate only arguments checking
    """

    import re
    import sys
    import logging

    from cars import prepare
    from cars import compute_dsm
    from cars import utils
    from cars import parameters as params
    from cars import configuration_correlator as corr_cfg

    # logging
    numeric_level = getattr(logging, args.loglevel.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    logging.basicConfig(
        level=numeric_level,
        datefmt='%y-%m-%d %H:%M:%S',
        format='%(asctime)s :: %(levelname)s :: %(message)s')

    # main(s) for each command
    if args.command == "prepare":
        # Check remaining arguments
        stop_now = False
        if not os.path.exists(args.injson):
            logging.critical('{} does not exist'.format(args.injson))
            stop_now = True
        if args.epi_step < 1:
            logging.critical('{} is an invalid value for --epi_step parameter \
            (should be > 1)'.format(args.epi_step))
            stop_now = True
        if args.disparity_margin < 0 or args.disparity_margin > 1:
            logging.critical(
                '{} is an invalid value for --disparity_margin parameter \
            (should be in range [0,1])'.format(
                    args.disparity_margin))
            stop_now = True
        if args.epipolar_error_upper_bound <= 0:
            logging.critical(
                '{} is an invalid value for --epipolar_error_upper_bound \
            parameter (should be > 0)'.format(
                    args.epipolar_error_upper_bound))
            stop_now = True
        if args.epipolar_error_maximum_bias < 0:
            logging.critical(
                '{} is an invalid value for --epipolar_error_maximum_bias \
            parameter (should be >= 0)'.format(
                    args.epipolar_error_maximum_bias))
            stop_now = True
        if args.nb_workers < 1:
            logging.critical(
                '{} is an invalid value for --nb_workers parameter \
            (should be > 0)'.format(
                    args.nb_workers))
            stop_now = True
        if not re.match(r'[0-9]{2}:[0-9]{2}:[0-9]{2}', args.walltime):
            logging.critical('{} is an invalid value for --walltime parameter \
            (should match HH:MM:SS)'.format(args.walltime))
            stop_now = True
        if args.elevation_delta_upper_bound <= args.elevation_delta_lower_bound:
            logging.critical(
                '--elevation_delta_lower_bound = {} is greater than \
            --elevation_delta_upper_bound = {}'.format(
                    args.elevation_delta_lower_bound,
                    args.elevation_delta_upper_bound))
            stop_now = True

        # If there are invalid parameters, stop now
        if stop_now:
            logging.critical(
                "Invalid parameters detected, please fix cars_cli \
            prepare command-line.")
            sys.exit(1)

        # Read input json file
        in_json = params.read_input_parameters(args.injson)

        if not check_inputs:
            prepare.run(
                in_json,
                args.outdir,
                epi_step=args.epi_step,
                disparity_margin=args.disparity_margin,
                epipolar_error_upper_bound=args.epipolar_error_upper_bound,
                epipolar_error_maximum_bias=args.epipolar_error_maximum_bias,
                elevation_delta_lower_bound=args.elevation_delta_lower_bound,
                elevation_delta_upper_bound=args.elevation_delta_upper_bound,
                mode=args.mode,
                nb_workers=args.nb_workers,
                walltime=args.walltime,
                check_inputs=args.check_inputs)

    elif args.command == "compute_dsm":
        # Check remaining arguments
        stop_now = False
        if len(args.injsons) == 0:
            logging.critical('At least one input json file is required')
            stop_now = True
        for f in args.injsons:
            if not os.path.exists(f):
                logging.critical('{} does not exist'.format(f))
                stop_now = True
        if args.sigma is not None and args.sigma < 0:
            logging.critical('{} is an invalid value for --sigma parameter \
            (should be >= 0)'.format(args.sigma))
            stop_now = True
        if args.dsm_radius < 0:
            logging.critical(
                '{} is an invalid value for --dsm_radius parameter \
            (should be >= 0)'.format(
                    args.dsm_radius))
            stop_now = True
        if args.resolution <= 0:
            logging.critical(
                '{} is an invalid value for --resolution parameter \
            (should be > 0)'.format(
                    args.resolution))
            stop_now = True
        if args.epsg is not None and args.epsg < 1:
            logging.critical('{} is an invalid value for --epsg parameter \
            (should be > 0)'.format(args.epsg))
            stop_now = True
        if args.corr_config is not None:
            if not os.path.exists(args.corr_config):
                logging.critical('{} does not exist'.format(args.corr_config))
                stop_now = True
        if args.nb_workers < 1:
            logging.critical(
                '{} is an invalid value for --nb_workers parameter \
            (should be > 0)'.format(
                    args.nb_workers))
            stop_now = True
        if not re.match(r'[0-9]{2}:[0-9]{2}:[0-9]{2}', args.walltime):
            logging.critical('{} is an invalid value for --walltime parameter \
            (should match HH:MM:SS)'.format(args.walltime))
            stop_now = True
        if args.max_elevation_offset is not None and args.min_elevation_offset is not None \
           and args.max_elevation_offset <= args.min_elevation_offset:
            logging.critical('--min_elevation_offset = {} is greater than \
            --max_elevation_offset = {}'.format(args.min_elevation_offset,
                                                args.max_elevation_offset))
            stop_now = True
        roi, stop_now = parse_roi_argument(args.roi, stop_now)

        # If there are invalid parameters, stop now
        if stop_now:
            logging.critical(
                "Invalid parameters detected, please fix cars_cli \
            compute_dsm command-line.")
            sys.exit(1)

        # Read input json files
        in_jsons = [params.read_preprocessing_content_file(
            f) for f in args.injsons]
        # Configure correlator
        corr_config = corr_cfg.configure_correlator(args.corr_config)

        if not check_inputs:
            compute_dsm.run(
                in_jsons,
                args.outdir,
                resolution=args.resolution,
                min_elevation_offset=args.min_elevation_offset,
                max_elevation_offset=args.max_elevation_offset,
                epsg=args.epsg,
                sigma=args.sigma,
                dsm_radius=args.dsm_radius,
                dsm_no_data=args.dsm_no_data,
                color_no_data=args.color_no_data,
                corr_config=corr_config,
                output_stats=args.output_stats,
                mode=args.mode,
                nb_workers=args.nb_workers,
                walltime=args.walltime,
                roi=roi,
                use_geoid_alt=args.use_geoid_as_alt_ref,
                snap_to_img1 = args.snap_to_left_image,
                align = args.align_with_lowres_dem,
                cloud_small_components_filter=not args.disable_cloud_small_components_filter,
                cloud_statistical_outliers_filter=not args.disable_cloud_statistical_outliers_filter,
                use_sec_disp=args.use_sec_disp
            )

    else:
        parser.print_help()
        sys.exit(1)


def entry_point():
    parser = cars_cli_parser()
    args = parser.parse_args()
    main_cli(args, parser)


if __name__ == '__main__':
    entry_point()
