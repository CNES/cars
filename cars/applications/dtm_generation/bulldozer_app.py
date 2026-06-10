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
This module contains the bulldozer dsm filling application class.
"""

import contextlib
import logging
import os
from pathlib import Path

import rasterio as rio
import yaml
from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm
from json_checker import Checker

import cars.orchestrator.orchestrator as ocht
from cars.applications.dem_generation.dem_generation_wrappers import (
    edit_transform,
)
from cars.core import preprocessing
from cars.orchestrator.cluster.log_wrapper import cars_profile

from .abstract_dtm_generation_app import DtmGeneration


class Bulldozer(DtmGeneration, short_name="bulldozer"):
    """
    Bulldozer app
    """

    def __init__(self, conf=None):
        """
        Init function of Bulldozer

        :param conf: configuration for Bulldozer
        :return: an application_to_use object
        """
        super().__init__(conf=conf)

        # check conf
        self.used_method = self.used_config["method"]
        self.save_intermediate_data = self.used_config["save_intermediate_data"]

    def check_conf(self, conf):

        # init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        # Overload conf
        overloaded_conf["method"] = conf.get("method", "bulldozer")

        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        rectification_schema = {
            "method": str,
            "save_intermediate_data": bool,
        }

        # Check conf
        checker = Checker(rectification_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    @cars_profile(name="Bulldozer filling")
    def run(  # pylint: disable=too-many-positional-arguments # noqa C901
        self,
        dsm_file,
        dump_dir,
        orchestrator=None,
        dsm_dir=None,
    ):
        """
        Run bulldozer to get the DTM

        roi_poly can any of these objects :
            - a list of Shapely Polygons
            - a Shapely Polygon
        """
        if orchestrator is None:
            orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )

        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        # create the config for the bulldozer execution
        bull_conf_path = os.path.join(
            os.path.dirname(__file__), "bulldozer_config/base_config.yaml"
        )
        with open(bull_conf_path, "r", encoding="utf8") as bull_conf_file:
            bull_conf = yaml.safe_load(bull_conf_file)

        bull_conf["dsm_path"] = dsm_file
        bull_conf["output_dir"] = os.path.join(dump_dir, "bulldozer")

        if orchestrator is not None:
            if (
                orchestrator.get_conf()["mode"] == "multiprocessing"
                or orchestrator.get_conf()["mode"] == "local_dask"
            ):
                bull_conf["nb_max_workers"] = orchestrator.get_conf()[
                    "nb_workers"
                ]

        bull_conf_path = os.path.join(dump_dir, "bulldozer_config.yaml")
        with open(bull_conf_path, "w", encoding="utf8") as bull_conf_file:
            yaml.dump(bull_conf, bull_conf_file)

        # Modify DSM for dtm to be used
        with rio.open(dsm_file) as in_dsm:
            dsm_tr = in_dsm.transform
            dsm_crs = in_dsm.crs
            dsm_bounds = in_dsm.bounds

        saved_transform = None
        if dsm_crs.is_geographic:
            xmin = dsm_bounds.left
            ymin = dsm_bounds.bottom
            utm_epsg = preprocessing.get_utm_zone_as_epsg_code(xmin, ymin)
            conversion_factor = preprocessing.get_conversion_factor(
                dsm_bounds, utm_epsg, dsm_crs.to_epsg()
            )
            resolution = dsm_tr.a * conversion_factor
            saved_transform = edit_transform(dsm_file, resolution=resolution)

        dtm_path = os.path.join(bull_conf["output_dir"], "dtm.tif")

        # Launch Bulldozer
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
                logging.info(
                    "Bulldozer failed on its first execution. Retrying"
                )
                # suppress prints in bulldozer by redirecting stdout&stderr
                with open(os.devnull, "w", encoding="utf8") as devnull:
                    with (
                        contextlib.redirect_stdout(devnull),
                        contextlib.redirect_stderr(devnull),
                    ):
                        dsm_to_dtm(bull_conf_path)
        except Exception:
            logging.warning(
                "Bulldozer failed on its second execution."
                + " The DSM could not be filled."
            )

        # Change back dsm and dtm to previous ref
        if saved_transform is not None:
            edit_transform(dtm_path, transform=saved_transform)
            edit_transform(dsm_file, transform=saved_transform)

        for element in Path(bull_conf["output_dir"]).iterdir():
            if element.name == "dtm.tif":
                os.replace(element, Path(dsm_dir) / element.name)
