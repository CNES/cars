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
            raise FileNotFoundError(geomodel + " does not exist")

    elif basename.endswith(".tif"):
        geomodel = os.path.splitext(absfilename)[0] + ".geom"
        if os.path.exists(geomodel) is False:
            raise FileNotFoundError(geomodel + " does not exist")
    else:
        raise ValueError(absfilename + " not supported")

    sensor["image"] = absfilename
    sensor["geomodel"] = geomodel

    return sensor


def pairdirname_to_pc(pairdirname):
    """
    Fill sensor dictionary according to an pair directory
    """
    sensor = {}

    abspairdirname = os.path.abspath(pairdirname)

    for coord in ["X", "Y", "Z"]:
        bandname = os.path.join(abspairdirname, "epi_pc_" + coord + ".tif")
        if os.path.isfile(bandname) is False:
            raise FileNotFoundError(bandname + " does not exist")
        sensor[coord.lower()] = bandname

    for extra in ["color"]:
        bandname = os.path.join(abspairdirname, "epi_pc_" + extra + ".tif")
        if os.path.isfile(bandname):
            sensor[extra] = bandname

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

    # check first input in list to determine pipeline
    if os.path.isfile(config["il"][0]):
        cars_config = {"inputs": {"sensors": {}}, "output": {}}
        pipeline_name = "sensors_to_dense_dsm"

        for idx, inputfilename in enumerate(config["il"]):
            cars_config["inputs"]["sensors"][str(idx)] = (
                inputfilename_to_sensor(inputfilename)
            )

        # pairing with first image as reference
        pairing = list(
            zip(  # noqa: B905
                ["0"] * (len(config["il"]) - 1),
                map(str, range(1, len(config["il"]))),
            )
        )

        cars_config["inputs"]["pairing"] = pairing

    else:
        cars_config = {"inputs": {"point_clouds": {}}, "output": {}}
        pipeline_name = "dense_point_clouds_to_dense_dsm"

        for idx, pairdirname in enumerate(config["il"]):
            cars_config["inputs"]["point_clouds"][str(idx)] = pairdirname_to_pc(
                pairdirname
            )

    cars_config["output"]["out_dir"] = config["out"]

    check = config["check"] if "check" in config.keys() else False
    full = config["full"] if "full" in config.keys() else False

    if check or full:
        # cars imports
        from cars.pipelines.pipeline import Pipeline

        used_pipeline = Pipeline(pipeline_name, cars_config, None)
        if full:
            cars_config = used_pipeline.used_conf

    print(json.dumps(cars_config, indent=4))


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
        metavar="input.{tif,XML} or pair_dir",
        help="Inputs list or Pairs directory list",
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

    parser.add_argument("--check", action="store_true", help="Check inputs")

    args = parser.parse_args()
    cars_starter(vars(args))


if __name__ == "__main__":
    cli()
