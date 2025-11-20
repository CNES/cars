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
cars-starter: helper to create configuration file
"""

# pylint: disable=import-outside-toplevel
# standard imports
import argparse
import json
import os

import yaml


def inputfilename_to_sensor(inputfilename):
    """
    Fill sensor dictionary according to an input filename
    """
    sensor = {}

    absfilename = os.path.abspath(inputfilename)
    dirname = os.path.dirname(absfilename)
    basename = os.path.basename(absfilename)

    if basename.startswith("DIM") and basename.endswith("XML"):
        geomodel = os.path.join(dirname, "RPC" + basename[3:])
        if os.path.exists(geomodel) is False:
            geomodel = None

    elif basename.endswith(".tif"):
        geomodel = os.path.splitext(absfilename)[0] + ".geom"
        if os.path.exists(geomodel) is False:
            geomodel = None
    else:
        raise ValueError(absfilename + " not supported")

    sensor["image"] = absfilename
    if geomodel is not None:
        sensor["geomodel"] = geomodel

    return sensor


def cars_starter(cli_params: dict = None, **kwargs) -> None:
    """
    Main fonction. Expects a dictionary from the CLI (cli_params)
    or directly the input parameters.
    """
    if cli_params and isinstance(cli_params, dict):
        config = cli_params
    else:
        params_name = set(kwargs.keys())
        required_params = {"il", "out"}
        missing_params = required_params - params_name
        if len(missing_params) > 0:
            raise ValueError(
                "The following parameters are required: {}".format(
                    ", ".join(list(missing_params))
                )
            )
        config = kwargs

    cars_config = {"input": {"sensors": {}}, "output": {}}
    pipeline_name = "default"

    for idx, inputfilename in enumerate(config["il"]):
        cars_config["input"]["sensors"][str(idx)] = inputfilename_to_sensor(
            inputfilename
        )

    # pairing with first image as reference
    pairing = list(
        zip(  # noqa: B905
            ["0"] * (len(config["il"]) - 1),
            map(str, range(1, len(config["il"]))),
        )
    )

    cars_config["input"]["pairing"] = pairing
    cars_config["output"]["directory"] = config["out"]

    check = config["check"] if "check" in config.keys() else False
    full = config["full"] if "full" in config.keys() else False

    if check or full:
        # cars imports
        from cars.pipelines.pipeline import Pipeline

        used_pipeline = Pipeline(pipeline_name, cars_config, None)
        if full:
            cars_config = used_pipeline.used_conf

    # output format handling
    output_format = config.get("format", "yaml")
    if output_format == "yaml":
        print(yaml.safe_dump(cars_config, sort_keys=False))
    elif output_format == "json":
        print(json.dumps(cars_config, indent=2))
    else:
        raise ValueError(
            "Invalid format: {}. Use 'json' or 'yaml'.".format(output_format)
        )


def cli():
    """
    Main cars-starter entrypoint (Command Line Interface)
    """
    parser = argparse.ArgumentParser(
        "cars-starter", description="Helper to create configuration file"
    )
    parser.add_argument(
        "-il",
        type=str,
        nargs="*",
        metavar="input.{tif,XML}",
        help="Input sensor list",
        required=True,
    )

    parser.add_argument(
        "-out",
        type=str,
        metavar="out_dir",
        help="Output directory",
        required=True,
    )

    parser.add_argument(
        "--full", action="store_true", help="Fill all default values"
    )

    parser.add_argument("--check", action="store_true", help="Check input")

    parser.add_argument(
        "--format",
        type=str,
        default="yaml",
        choices=["json", "yaml"],
        help="Output format (json or yaml). Default: yaml",
    )

    args = parser.parse_args()
    cars_starter(vars(args))


if __name__ == "__main__":
    cli()
