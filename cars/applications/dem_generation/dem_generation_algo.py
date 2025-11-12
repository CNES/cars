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
this module contains tools for the dem generation
"""

import contextlib
import logging

# Standard imports
import os

# Third party imports
import yaml
from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm


def launch_bulldozer(
    input_dem,
    output_dir,
    cars_orchestrator,
    max_object_size,
):
    """
    Launch bulldozer on a DEM to smooth it

    :param input_dem: path of DEM to reverse
    :type input_dem: str
    :param output_dir: directory where bulldozer output is dumped
    :type output_dir: str
    :param cars_orchestrator: orchestrator of the whole pipeline
                              (used to get number of workers)
    :type cars_orchestrator: Orchestrator
    :param max_object_size: bulldozer parameter "max_object_size"
    :type max_object_size: int
    """
    bull_conf_path = os.path.join(
        os.path.dirname(__file__), "bulldozer_config/base_config.yaml"
    )
    with open(bull_conf_path, "r", encoding="utf8") as bull_conf_file:
        bull_conf = yaml.safe_load(bull_conf_file)

    bull_conf["dsm_path"] = input_dem
    bull_conf["output_dir"] = output_dir
    if cars_orchestrator is not None:
        if (
            cars_orchestrator.get_conf()["mode"] == "multiprocessing"
            or cars_orchestrator.get_conf()["mode"] == "local_dask"
        ):
            bull_conf["nb_max_workers"] = cars_orchestrator.get_conf()[
                "nb_workers"
            ]
    else:
        bull_conf["nb_max_workers"] = 4
    bull_conf["max_object_size"] = max_object_size

    os.makedirs(output_dir, exist_ok=True)
    bull_conf_path = os.path.join(output_dir, "bulldozer_config.yaml")
    with open(bull_conf_path, "w", encoding="utf8") as bull_conf_file:
        yaml.dump(bull_conf, bull_conf_file)

    output_dem = os.path.join(bull_conf["output_dir"], "dtm.tif")

    try:
        try:
            # suppress prints in bulldozer by redirecting stdout&stderr
            with open(os.devnull, "w", encoding="utf8") as devnull:
                with (
                    contextlib.redirect_stdout(devnull),
                    contextlib.redirect_stderr(devnull),
                ):
                    dsm_to_dtm(bull_conf_path)
        except Exception:
            logging.info("Bulldozer failed on its first execution. Retrying")
            # suppress prints in bulldozer by redirecting stdout&stderr
            with open(os.devnull, "w", encoding="utf8") as devnull:
                with (
                    contextlib.redirect_stdout(devnull),
                    contextlib.redirect_stderr(devnull),
                ):
                    dsm_to_dtm(bull_conf_path)
    except Exception:
        logging.error(
            "Bulldozer failed on its second execution."
            + " The DSM could not be smoothed."
        )
        output_dem = None

    return output_dem
