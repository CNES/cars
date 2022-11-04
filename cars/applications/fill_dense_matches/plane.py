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
this module contains the fill_disp application class.
"""

# Standard imports
import logging
from typing import Dict, Tuple

import xarray as xr

# Third party imports
from json_checker import Checker, Or

import cars.orchestrator.orchestrator as ocht

# CARS imports
from cars.applications.fill_dense_matches import FillDisp
from cars.applications.fill_dense_matches import fill_disp_constants as fd_cst
from cars.applications.fill_dense_matches import fill_disp_tools as fd_tools


class PlaneFill(FillDisp, short_name=["plane"]):  # pylint: disable=R0903
    """
    Fill invalid area in disparity map using plane method
    """

    def __init__(self, conf=None):
        """
        Init function of FillDisp

        :param conf: configuration for filling
        :return: a application_to_use object
        """

        # Check conf
        checked_conf = self.check_conf(conf)
        # used_config used for printing config
        self.used_config = checked_conf

        # check conf
        self.used_method = checked_conf["method"]
        self.interpolation_type = checked_conf[fd_cst.INTERP_TYPE]
        self.interpolation_method = checked_conf[fd_cst.INTERP_METHOD]
        self.max_search_distance = checked_conf[fd_cst.MAX_DIST]
        self.smoothing_iterations = checked_conf[fd_cst.SMOOTH_IT]
        self.ignore_nodata_at_disp_mask_borders = checked_conf[
            fd_cst.IGNORE_NODATA
        ]
        self.ignore_zero_fill_disp_mask_values = checked_conf[
            fd_cst.IGNORE_ZERO
        ]
        self.ignore_extrema_disp_values = checked_conf[fd_cst.IGNORE_EXTREMA]
        self.nb_pix = checked_conf["nb_pix"]
        self.percent_to_erode = checked_conf["percent_to_erode"]

        # init orchestrator
        self.orchestrator = None

    def check_conf(self, conf):
        """
        Check configuration

        :param conf: configuration to check
        :type conf: dict
        :return: overloaded configuration
        :rtype: dict

        """

        # init conf
        if conf is not None:
            overloaded_conf = conf.copy()
        else:
            conf = {}
            overloaded_conf = {}

        # Overload conf
        overloaded_conf["method"] = conf.get("method", "plane")
        overloaded_conf[fd_cst.INTERP_TYPE] = conf.get(
            fd_cst.INTERP_TYPE, "pandora"
        )
        overloaded_conf[fd_cst.INTERP_METHOD] = conf.get(
            fd_cst.INTERP_METHOD, "mc_cnn"
        )
        overloaded_conf[fd_cst.MAX_DIST] = conf.get(fd_cst.MAX_DIST, 100)
        overloaded_conf[fd_cst.SMOOTH_IT] = conf.get(fd_cst.SMOOTH_IT, 1)
        overloaded_conf[fd_cst.IGNORE_NODATA] = conf.get(
            fd_cst.IGNORE_NODATA, True
        )
        overloaded_conf[fd_cst.IGNORE_ZERO] = conf.get(fd_cst.IGNORE_ZERO, True)
        overloaded_conf[fd_cst.IGNORE_EXTREMA] = conf.get(
            fd_cst.IGNORE_EXTREMA, True
        )
        overloaded_conf["nb_pix"] = conf.get("nb_pix", 20)
        overloaded_conf["percent_to_erode"] = conf.get("percent_to_erode", 0.2)

        application_schema = {
            "method": str,
            "interpolation_type": Or(None, str),
            "interpolation_method": Or(None, str),
            "max_search_distance": Or(None, int),
            "smoothing_iterations": Or(None, int),
            "ignore_nodata_at_disp_mask_borders": bool,
            "ignore_zero_fill_disp_mask_values": bool,
            "ignore_extrema_disp_values": bool,
            "nb_pix": Or(None, int),
            "percent_to_erode": Or(None, float),
        }

        # Check conf
        checker = Checker(application_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(
        self,
        roi_mapping,
        orchestrator=None,
    ):
        """
        Run Refill application using plane method.

        TODO adapt to CarsDataset, now list of tiles
        """

        # Default orchestrator
        if orchestrator is None:
            # Create defaut sequential orchestrator for current application
            # be awere, no out_json will be shared between orchestrators
            # No files saved
            self.orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )
        else:
            self.orchestrator = orchestrator

        interp_options = {
            "type": self.interpolation_type,
            "method": self.interpolation_method,
            "smoothing_iterations": self.smoothing_iterations,
            "max_search_distance": self.max_search_distance,
        }

        if isinstance(roi_mapping, dict):
            # Fill masked areas of disparity map
            for __, data in enumerate(roi_mapping.values()):
                # Compute disparity
                self.orchestrator.cluster.create_task(
                    fill_disp_using_plane, nout=0
                )(
                    data["dataset"],
                    self.ignore_nodata_at_disp_mask_borders,
                    self.ignore_zero_fill_disp_mask_values,
                    self.ignore_extrema_disp_values,
                    self.nb_pix,
                    self.percent_to_erode,
                    interp_options,
                )
        else:
            logging.error(
                "FillDisp application doesn't support this input data format"
            )


def fill_disp_using_plane(
    left_disp_map: xr.Dataset,
    ignore_nodata: bool,
    ignore_zero_fill: bool,
    ignore_extrema: bool,
    nb_pix: int,
    percent_to_erode: float,
    interp_options: dict,
) -> Dict[str, Tuple[xr.Dataset, xr.Dataset]]:
    """
    TODO

    :param left_disp_map: TODO
    :type left_disp_map:
    :param ignore_nodata: TODO
    :type ignore_nodata: bool
    :param ignore_zero_fill: TODO
    :type ignore_zero_fill: bool
    :param ignore_extrema: TODO
    :type ignore_extrema: bool
    :param nb_pix: TODO
    :type nb_pix: int
    :param percent_to_erode: TODO
    :type percent_to_erode: float
    :param interp_options: interp_options
    :type interp_options: dict
    :return: overloaded configuration
    :rtype: dict

    """
    border_region = fd_tools.fill_central_area_using_plane(
        left_disp_map,
        ignore_nodata,
        ignore_zero_fill,
        ignore_extrema,
        nb_pix,
        percent_to_erode,
    )

    fd_tools.fill_area_borders_using_interpolation(
        left_disp_map,
        border_region,
        interp_options,
    )
