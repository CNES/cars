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
import json
import logging
import os
import re
import sys

# CARS imports
from cars import __version__
from cars.core import cars_logging


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
        """Redefine assignment"""
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


def cars_parser() -> CarsArgumentParser:
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

    parser.add_argument("conf", type=str, help="Inputs Configuration File")

    parser.add_argument(
        "--loglevel",
        default="PROGRESS",
        choices=("DEBUG", "INFO", "PROGRESS", "WARNING", "ERROR", "CRITICAL"),
        help="Logger level (default: WARNING. Should be one of "
        "(DEBUG, INFO, PROGRESS, WARNING, ERROR, CRITICAL)",
    )

    # General arguments at first level
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )

    return parser


def main_cli(args, dry_run=False):  # noqa: C901
    """
    Main for command line management

    :param dry_run: activate only arguments checking
    """
    # TODO : refactor in order to avoid a slow argparse
    # Don't move the local function imports for now

    # Change stdout to clean (Os) OTB output from image_envelope app.
    original_stdout = sys.stdout
    sys.stdout = StreamCapture(sys.stdout, r"(0s)")

    # Logging configuration with args Loglevel
    loglevel = getattr(args, "loglevel", "PROGRESS").upper()
    cars_logging.create(loglevel)
    logging.debug("Show argparse arguments: {}".format(args))

    # Force the use of OpenMP in numba
    os.environ["NUMBA_THREADING_LAYER"] = "omp"

    # Force the use of CARS dask configuration
    dask_config_path = os.path.join(
        os.path.dirname(__file__), "orchestrator", "cluster", "dask_config"
    )
    if not os.path.isdir(dask_config_path):
        raise NotADirectoryError("Wrong dask config path")
    os.environ["DASK_CONFIG"] = str(dask_config_path)

    # Main try/except to catch all program exceptions
    from cars.pipelines.pipeline import Pipeline

    try:
        # main(s) for each command
        if args.conf is not None:
            # Transform conf file to dict
            with open(args.conf, "r", encoding="utf8") as fstream:
                config = json.load(fstream)

            config_json_dir = os.path.abspath(os.path.dirname(args.conf))

            pipeline_name = config.get("pipeline", "sensors_to_dense_dsm")
            # Generate pipeline and check conf
            cars_logging.add_progress_message("Check configuration...")
            used_pipeline = Pipeline(pipeline_name, config, config_json_dir)
            cars_logging.add_progress_message("CARS pipeline is started.")
            if not dry_run:
                # run pipeline
                used_pipeline.run()
            cars_logging.add_progress_message(
                "CARS has successfully completed the pipeline."
            )
        else:
            raise SystemExit("CARS wrong subcommand. Use cars --help")
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
    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])
    main_cli(args)


if __name__ == "__main__":
    main()
