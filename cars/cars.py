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
"""
Main CARS Command Line Interface
user main argparse wrapper to CARS 3D pipelines submodules
"""

# Standard imports
# TODO refactor but keep local functions for performance and remove pylint
# pylint: disable=import-outside-toplevel
import argparse
import logging
import os
import re
import sys
from typing import List, Tuple

# Third party imports
import argcomplete

# CARS imports
from cars import __version__


class StreamCapture:
    """Filter stream (for stdout) with a re pattern
    From https://stackoverflow.com/a/63662744
    """

    def __init__(self, stream, re_pattern):
        """StreamCapture constructor: add pattern, triggered parameters"""
        self.stream = stream
        self.pattern = (
            re.compile(re_pattern)
            if isinstance(re_pattern, str)
            else re_pattern
        )
        self.triggered = False

    def __getattr__(self, attr_name):
        """Redefine assignement"""
        return getattr(self.stream, attr_name)

    def write(self, data):
        """Change write function of stream and deals \n for loops"""
        if data == "\n" and self.triggered:
            self.triggered = False
        else:
            if self.pattern.search(data) is None:
                # Pattern not found, write normally
                self.stream.write(data)
                self.stream.flush()
            else:
                # caught pattern to filter, no writing.
                self.triggered = True

    def flush(self):
        self.stream.flush()


class CarsArgumentParser(argparse.ArgumentParser):
    """
    ArgumentParser class adaptation for CARS
    """

    def convert_arg_line_to_args(self, arg_line):
        """
        Redefine herited function to accept one line argument parser in @file
        from fromfile_prefix_chars file argument
        https://docs.python.org/dev/library/argparse.html
        """
        return arg_line.split()


def cars_parser() -> argparse.ArgumentParser:
    """
    Main CLI argparse parser function
    It builds argparse objects and constructs CLI interfaces parameters.

    :return: CARS arparse CLI interface object
    """
    # Create cars cli parser from argparse
    # use @file to use a file containing parameters
    parser = CarsArgumentParser(
        "cars",
        description="CARS: CNES Algorithms to Reconstruct Surface",
        fromfile_prefix_chars="@",
    )

    # General arguments at first level
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )

    # Create subcommand parser for prepare and compute_dsm
    subparsers = parser.add_subparsers(dest="command")

    # Prepare subcommand

    # Prepare subparser creation
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Preparation for compute_dsm "
        "producing stereo-rectification grid as well "
        "as an estimate of the disparity to explore.",
    )

    # Prepare arguments

    # Mandatories (in a specific argparse group)
    prepare_parser_mandatory = prepare_parser.add_argument_group(
        "mandatory arguments"
    )
    prepare_parser_mandatory.add_argument(
        "-i", "--injson", required=True, type=str, help="Input json file"
    )
    prepare_parser_mandatory.add_argument(
        "-o", "--outdir", required=True, type=str, help="Output directory"
    )

    # Optionals
    prepare_parser.add_argument(
        "--epi_step",
        type=int,
        default=30,
        help="Step of the deformation grid in nb. of pixels "
        "(default: 30, should be > 1)",
    )
    prepare_parser.add_argument(
        "--disparity_margin",
        type=float,
        default=0.02,
        help="Add a margin to min and max disparity as percent "
        "of the disparity range "
        "(default: 0.02, should be in range [0,1])",
    )
    prepare_parser.add_argument(
        "--epipolar_error_upper_bound",
        type=float,
        default=10.0,
        help="Expected upper bound for epipolar error in pixels "
        "(default: 10, should be > 0)",
    )
    prepare_parser.add_argument(
        "--epipolar_error_maximum_bias",
        type=float,
        default=0.0,
        help="Maximum bias for epipolar error in pixels "
        "(default: 0, should be >= 0)",
    )
    prepare_parser.add_argument(
        "--elevation_delta_lower_bound",
        type=float,
        default=-1000.0,
        help="Expected lower bound for elevation delta "
        "with respect to input low resolution DTM in meters "
        "(default: -1000)",
    )
    prepare_parser.add_argument(
        "--elevation_delta_upper_bound",
        type=float,
        default=1000.0,
        help="Expected upper bound for elevation delta "
        "with respect to input low resolution DTM in meters "
        "(default: 1000)",
    )
    prepare_parser.add_argument(
        "--mode",
        default="local_dask",
        choices=("pbs_dask", "local_dask"),
        help="Parallelization mode (default: local_dask)",
    )
    prepare_parser.add_argument(
        "--nb_workers",
        type=int,
        default=2,
        help="Number of workers (default: 2, should be > 0)",
    )
    prepare_parser.add_argument(
        "--walltime",
        default="00:59:00",
        help="Walltime for one worker (default: 00:59:00). "
        "Should be formatted as HH:MM:SS)",
    )
    prepare_parser.add_argument(
        "--check_inputs", action="store_true", help="Check inputs consistency"
    )
    prepare_parser.add_argument(
        "--loglevel",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logger level (default: INFO. Should be one of "
        "(DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # Compute_dsm subcommand

    # Prepare subparser creation
    compute_dsm_parser = subparsers.add_parser(
        "compute_dsm",
        help="Tile-based, concurent resampling "
        "in epipolar geometry, disparity "
        "estimation, triangulation and rasterization",
    )

    # Compute_dsm arguments

    # Mandatories (in a specific argparse group)
    compute_dsm_parser_mandatory = compute_dsm_parser.add_argument_group(
        "mandatory arguments"
    )
    compute_dsm_parser_mandatory.add_argument(
        "-i", "--injsons", required=True, help="Input json files", nargs="*"
    )
    compute_dsm_parser_mandatory.add_argument(
        "-o", "--outdir", required=True, type=str, help="Output directory"
    )

    # Optionals
    compute_dsm_parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Sigma for rasterization in fraction of pixels "
        "(default: None, should be >= 0)",
    )
    compute_dsm_parser.add_argument(
        "--dsm_radius",
        type=int,
        default=1,
        help="Radius for rasterization in pixels "
        "(default: 1, should be >= 0)",
    )
    compute_dsm_parser.add_argument(
        "--resolution",
        type=float,
        default=0.5,
        help="Digital Surface Model resolution "
        "(default: 0.5, should be > 0)",
    )
    compute_dsm_parser.add_argument(
        "--epsg",
        type=int,
        default=None,
        help="EPSG code (default: None, should be > 0)",
    )
    compute_dsm_roi_group = compute_dsm_parser.add_mutually_exclusive_group()
    compute_dsm_roi_group.add_argument(
        "--roi_bbox",
        type=float,
        default=None,
        nargs=4,
        help="DSM ROI in final projection [xmin ymin xmax ymax] "
        "(it has to be in final projection)",
    )
    compute_dsm_roi_group.add_argument(
        "--roi_file",
        type=str,
        default=None,
        help="DSM ROI file (vector file or image which "
        "footprint will be taken as ROI).",
    )
    compute_dsm_parser.add_argument(
        "--dsm_no_data",
        type=int,
        default=-32768,
        help="No data value to use in the final DSM file (default: -32768)",
    )
    compute_dsm_parser.add_argument(
        "--color_no_data",
        type=int,
        default=0,
        help="No data value to use in the final color image (default: 0)",
    )
    compute_dsm_parser.add_argument(
        "--msk_no_data",
        help="No data value to use " "in the final mask image (default: 65535)",
        type=int,
        default=65535,
    )
    compute_dsm_parser.add_argument(
        "--corr_config",
        default=None,
        type=str,
        help="Correlator config (json file)",
    )
    compute_dsm_parser.add_argument(
        "--min_elevation_offset",
        type=float,
        default=None,
        help="Override minimum disparity from prepare step "
        "with this offset in meters",
    )
    compute_dsm_parser.add_argument(
        "--max_elevation_offset",
        type=float,
        default=None,
        help="Override maximum disparity from prepare step "
        "with this offset in meters",
    )
    compute_dsm_parser.add_argument(
        "--output_stats",
        action="store_true",
        help="Outputs dsm as a netCDF file embedding quality statistics.",
    )
    compute_dsm_parser.add_argument(
        "--use_geoid_as_alt_ref",
        action="store_true",
        default=False,
        help="Use geoid grid as altimetric reference.",
    )
    compute_dsm_parser.add_argument(
        "--use_sec_disp",
        action="store_true",
        help="Use the points cloud"
        "Generated from the secondary disparity map.",
    )
    compute_dsm_parser.add_argument(
        "--snap_to_left_image",
        action="store_true",
        default=False,
        help="This mode can be used if all pairs share the same left image. "
        "It will then modify lines of sights of secondary images "
        "so that they all cross those of the reference image.",
    )
    compute_dsm_parser.add_argument(
        "--align_with_lowres_dem",
        action="store_true",
        default=False,
        help="If this mode is used, during triangulation, "
        "points will be corrected using the estimated correction "
        "from the prepare step in order to align 3D points "
        "with the low resolution initial DEM.",
    )
    compute_dsm_parser.add_argument(
        "--disable_cloud_small_components_filter",
        action="store_true",
        help="This mode deactivates the points cloud "
        "filtering of small components.",
    )
    compute_dsm_parser.add_argument(
        "--disable_cloud_statistical_outliers_filter",
        action="store_true",
        help="This mode deactivates the points cloud "
        "filtering of statistical outliers.",
    )
    compute_dsm_parser.add_argument(
        "--mode",
        default="local_dask",
        choices=("pbs_dask", "local_dask", "mp"),
        help="Parallelization mode (default: local_dask) ",
    )
    compute_dsm_parser.add_argument(
        "--nb_workers",
        type=int,
        default=2,
        help="Number of workers (default: 2, should be > 0)",
    )
    compute_dsm_parser.add_argument(
        "--walltime",
        default="00:59:00",
        help="Walltime for one worker (default: 00:59:00). "
        "Should be formatted as HH:MM:SS)",
    )
    compute_dsm_parser.add_argument(
        "--loglevel",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logger level (default: INFO. Should be one of "
        "(DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # autocomplete
    argcomplete.autocomplete(parser)

    return parser


def parse_roi_file(
    arg_roi_file: str, stop_now: bool
) -> Tuple[List[float], int]:
    """
    Parse ROI file argument and generate bounding box


    :param arg_roi_file : ROI file argument
    :param stop_now: Argument check
    :return: ROI Bounding box + EPSG code : xmin, ymin, xmax, ymax, epsg_code
    :rtype: Tuple with array of 4 floats and int
    """
    # TODO : refactor in order to avoid a slow argparse
    # Don't move the local function imports for now

    # Third party imports
    import rasterio

    # CARS imports
    from cars.core import inputs

    # Declare output
    roi = None

    _, extension = os.path.splitext(arg_roi_file)

    # test file existence
    if not os.path.exists(arg_roi_file):
        logging.error("File {} does not exist".format(arg_roi_file))
        stop_now = True
    else:
        # if it is a vector file
        if extension in [".gpkg", ".shp", ".kml"]:
            roi_poly, roi_epsg = inputs.read_vector(arg_roi_file)
            roi = (roi_poly.bounds, roi_epsg)

        # if not, it is an image
        elif inputs.rasterio_can_open(arg_roi_file):
            data = rasterio.open(arg_roi_file)
            xmin = min(data.bounds.left, data.bounds.right)
            ymin = min(data.bounds.bottom, data.bounds.top)
            xmax = max(data.bounds.left, data.bounds.right)
            ymax = max(data.bounds.bottom, data.bounds.top)

            try:
                roi_epsg = data.crs.to_epsg()
                roi = ([xmin, ymin, xmax, ymax], roi_epsg)
            except AttributeError as error:
                logging.error("ROI EPSG code {} not readable".format(error))
                stop_now = True

        else:
            logging.error(
                "ROI file {} has an unsupported format".format(arg_roi_file)
            )
            stop_now = True

    return roi, stop_now


def run_prepare(args, dry_run=False):  # noqa: C901
    """
    Local function for running prepare pipeline from CLI
    :param args: arguments for prepare pipeline
    """
    # TODO : refactor in order to avoid a slow argparse
    # Don't move the local function imports for now

    # CARS imports
    from cars.conf import input_parameters
    from cars.pipelines import prepare

    # Check remaining arguments
    stop_now = False
    if not os.path.exists(args.injson):
        logging.error("File {} does not exist".format(args.injson))
        stop_now = True
    if args.epi_step < 1:
        logging.error(
            "{} is an invalid value for --epi_step parameter \
        (should be > 1)".format(
                args.epi_step
            )
        )
        stop_now = True
    if args.disparity_margin < 0 or args.disparity_margin > 1:
        logging.error(
            "{} is an invalid value for --disparity_margin parameter \
        (should be in range [0,1])".format(
                args.disparity_margin
            )
        )
        stop_now = True
    if args.epipolar_error_upper_bound <= 0:
        logging.error(
            "{} is an invalid value for --epipolar_error_upper_bound \
        parameter (should be > 0)".format(
                args.epipolar_error_upper_bound
            )
        )
        stop_now = True
    if args.epipolar_error_maximum_bias < 0:
        logging.error(
            "{} is an invalid value for --epipolar_error_maximum_bias \
        parameter (should be >= 0)".format(
                args.epipolar_error_maximum_bias
            )
        )
        stop_now = True
    if args.nb_workers < 1:
        logging.error(
            "{} is an invalid value for --nb_workers parameter \
        (should be > 0)".format(
                args.nb_workers
            )
        )
        stop_now = True
    if not re.match(r"[0-9]{2}:[0-9]{2}:[0-9]{2}", args.walltime):
        logging.error(
            "{} is an invalid value for --walltime parameter \
        (should match HH:MM:SS)".format(
                args.walltime
            )
        )
        stop_now = True
    if args.elevation_delta_upper_bound <= args.elevation_delta_lower_bound:
        logging.error(
            "--elevation_delta_lower_bound = {} is greater than \
        --elevation_delta_upper_bound = {}".format(
                args.elevation_delta_lower_bound,
                args.elevation_delta_upper_bound,
            )
        )
        stop_now = True

    # If there are invalid parameters, stop now
    if stop_now:
        raise SystemExit(
            "Invalid parameters detected, please fix cars prepare "
            "command-line."
        )

    # Read input json file
    in_json = input_parameters.read_input_parameters(args.injson)

    if not dry_run:
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
            check_inputs=args.check_inputs,
        )


def run_compute_dsm(args, dry_run=False):  # noqa: C901
    """
    Local function for running compute_dsm pipeline from CLI
    :param args: arguments for compute_dsm pipeline
    """
    # TODO : refactor in order to avoid a slow argparse
    # Don't move the local function imports for now

    # CARS imports
    from cars.conf import output_prepare
    from cars.pipelines import compute_dsm
    from cars.plugins.matching.correlator_configuration import corr_conf

    # Check remaining arguments
    stop_now = False
    if len(args.injsons) == 0:
        logging.error("One input json file is at least required")
        stop_now = True
    for json_file in args.injsons:
        if not os.path.exists(json_file):
            logging.error("File {} does not exist".format(json_file))
            stop_now = True
    if args.sigma is not None and args.sigma < 0:
        logging.error(
            "{} is an invalid value for --sigma parameter \
        (should be >= 0)".format(
                args.sigma
            )
        )
        stop_now = True
    if args.dsm_radius < 0:
        logging.error(
            "{} is an invalid value for --dsm_radius parameter \
        (should be >= 0)".format(
                args.dsm_radius
            )
        )
        stop_now = True
    if args.resolution <= 0:
        logging.error(
            "{} is an invalid value for --resolution parameter \
        (should be > 0)".format(
                args.resolution
            )
        )
        stop_now = True
    if args.epsg is not None and args.epsg < 1:
        logging.error(
            "{} is an invalid value for --epsg parameter \
        (should be > 0)".format(
                args.epsg
            )
        )
        stop_now = True
    if args.corr_config is not None:
        if not os.path.exists(args.corr_config):
            logging.error("File {} does not exist".format(args.corr_config))
            stop_now = True
    if args.nb_workers < 1:
        logging.error(
            "{} is an invalid value for --nb_workers parameter \
        (should be > 0)".format(
                args.nb_workers
            )
        )
        stop_now = True
    if not re.match(r"[0-9]{2}:[0-9]{2}:[0-9]{2}", args.walltime):
        logging.error(
            "{} is an invalid value for --walltime parameter \
        (should match HH:MM:SS)".format(
                args.walltime
            )
        )
        stop_now = True
    if (
        args.max_elevation_offset is not None
        and args.min_elevation_offset is not None
        and args.max_elevation_offset <= args.min_elevation_offset
    ):
        logging.error(
            "--min_elevation_offset = {} is greater than \
        --max_elevation_offset = {}".format(
                args.min_elevation_offset, args.max_elevation_offset
            )
        )
        stop_now = True

    # By default roi = None if no roi mutually exclusive options
    roi = None
    if args.roi_bbox is not None:
        # if roi_bbox defined, roi = 4 floats bounding box + EPSG code=None
        roi = (args.roi_bbox, None)
    if args.roi_file is not None:
        # If roi_file is defined, generate bouding box roi
        roi, stop_now = parse_roi_file(args.roi_file, stop_now)

    # If there are invalid parameters, stop now
    if stop_now:
        raise SystemExit(
            "Invalid parameters detected, please fix cars "
            "compute_dsm command-line."
        )

    # Read input json files
    in_jsons = [
        output_prepare.read_preprocessing_content_file(f) for f in args.injsons
    ]
    # Configure correlator
    corr_config = corr_conf.configure_correlator(args.corr_config)

    if not dry_run:
        # Prepare options not tested in test_cars.py

        # Inverse disable_cloud_small_components_filter
        small_components = not args.disable_cloud_small_components_filter
        # Inverse disable_cloud_statistical_outliers_filter
        stat_outliers = not args.disable_cloud_statistical_outliers_filter

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
            msk_no_data=args.msk_no_data,
            corr_config=corr_config,
            output_stats=args.output_stats,
            mode=args.mode,
            nb_workers=args.nb_workers,
            walltime=args.walltime,
            roi=roi,
            use_geoid_alt=args.use_geoid_as_alt_ref,
            snap_to_img1=args.snap_to_left_image,
            align=args.align_with_lowres_dem,
            cloud_small_components_filter=small_components,
            cloud_statistical_outliers_filter=stat_outliers,
            use_sec_disp=args.use_sec_disp,
        )


def main_cli(args, parser, dry_run=False):  # noqa: C901
    """
    Main for command line management

    :param dry_run: activate only arguments checking
    """
    # TODO : refactor in order to avoid a slow argparse
    # Don't move the local function imports for now

    # CARS imports
    from cars.conf.log_conf import setup_log

    # Change stdout to clean (Os) OTB output from image_envelope app.
    original_stdout = sys.stdout
    sys.stdout = StreamCapture(sys.stdout, r"(0s)")

    # Logging configuration with args Loglevel
    setup_log(args.loglevel.upper())

    # Debug argparse show args
    logging.debug("Show argparse arguments: {}".format(args))

    # import tbb test and force numba library tbb | omp
    from cars.cluster.tbb import check_tbb_installed

    numba_threading_layer = "omp"
    if not dry_run:
        if check_tbb_installed() and args.mode == "mp":
            numba_threading_layer = "tbb"
    os.environ["NUMBA_THREADING_LAYER"] = numba_threading_layer

    # Main try/except to catch all program exceptions
    try:
        # main(s) for each command
        if args.command is not None:
            if args.command == "prepare":
                run_prepare(args, dry_run)
            elif args.command == "compute_dsm":
                run_compute_dsm(args, dry_run)
            else:
                raise SystemExit("CARS launched with wrong subcommand")
        else:
            parser.print_help()
    except BaseException:
        # Catch all exceptions, show debug traceback and exit
        logging.exception("CARS terminated with following error")
        sys.exit(1)
    finally:
        # Go back to original stdout
        sys.stdout = original_stdout


def main():
    """
    Main initial cars cli entry point
    Configure and launch parser before main_cli function
    """
    # CARS parser
    parser = cars_parser()
    args = parser.parse_args()
    main_cli(args, parser)


if __name__ == "__main__":
    main()
