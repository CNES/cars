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
import sys

# CARS imports
from cars import __version__
from cars.core import cars_logging
from cars.orchestrator.cluster import log_wrapper


def cars_parser() -> argparse.ArgumentParser:
    """
    Main CLI argparse parser function
    It builds argparse objects and constructs CLI interfaces parameters.

    :return: CARS arparse CLI interface object
    """
    # Create cars cli parser from argparse
    # use @file to use a file containing parameters
    parser = argparse.ArgumentParser(
        "cars", description="CARS: CNES Algorithms to Reconstruct Surface"
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
        # Transform conf file to dict
        with open(args.conf, "r", encoding="utf8") as fstream:
            config = json.load(fstream)

        config_json_dir = os.path.abspath(os.path.dirname(args.conf))
        pipeline_name = config.get(
            "pipeline", "sensors_to_dense_dsm_no_merging"
        )

        # Logging configuration with args Loglevel
        loglevel = getattr(args, "loglevel", "PROGRESS").upper()
        out_dir = config["output"]["out_dir"]
        cars_logging.setup_logging(
            loglevel, out_dir=out_dir, pipeline=pipeline_name
        )
        logging.debug("Show argparse arguments: {}".format(args))

        # Generate pipeline and check conf
        cars_logging.add_progress_message("Check configuration...")
        used_pipeline = Pipeline(pipeline_name, config, config_json_dir)
        cars_logging.add_progress_message("CARS pipeline is started.")
        if not dry_run:
            # run pipeline
            used_pipeline.run()

        # Generate summary of tasks
        log_wrapper.generate_summary(out_dir, used_pipeline.used_conf)
        cars_logging.add_progress_message(
            "CARS has successfully completed the pipeline."
        )

    except BaseException:
        # Catch all exceptions, show debug traceback and exit
        logging.exception("CARS terminated with following error")
        sys.exit(1)


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
