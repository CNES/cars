#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
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
This module contains the PANDORA2D sparse matching method implementation.
"""

import collections
import copy
import logging

import numpy as np
import pandas
import pandora2d
import xarray as xr
from json_checker import Checker
from pandora2d.check_configuration import check_datasets
from pandora2d.memory_estimation import estimate_total_consumption
from pandora2d.state_machine import Pandora2DMachine
from pandora.check_configuration import check_disparities_from_dataset

from cars.applications.dense_matching import dense_matching_algo as dm_algo
from cars.applications.sparse_matching import (
    sparse_matching_wrappers as sm_wrap,
)
from cars.applications.sparse_matching.loaders.pandora2d_loader import (
    Pandora2DLoader,
)
from cars.applications.sparse_matching.methods import (
    abstract_sparse_matching_method as asmm,
)
from cars.core import constants as cst
from cars.data_structures import cars_dataset

AbstractSparseMatchingMethod = asmm.AbstractSparseMatchingMethod


def is_positive(value):
    """
    Check if a value is positive.

    :param value: value to check
    :type value: int or float
    :return: True if value is positive, False otherwise
    :rtype: bool
    """
    return value > 0


class Pandora2DSparseMethod(
    AbstractSparseMatchingMethod, short_name=["pandora2d"]
):
    """
    Implementation of Pandora2d as a sparse matching method.
    """

    def __init__(self, conf):
        super().__init__(conf=conf)

        self.schema = {
            "method": str,
            "conf_to_use": str,
            "step": list,
            "loader_conf": dict,
            "loader": str,
            "used_band": str,
            "threshold_disp_range_to_borders": bool,
            "tile_width": int,
            "tile_height": int,
        }
        self.loader = None

        self.used_config = self.check_conf(conf)

        self.method = self.used_config["method"]
        self.conf_to_use = self.used_config["conf_to_use"]
        self.step = self.used_config["step"]
        self.loader_conf = self.used_config["loader_conf"]
        self.used_band = self.used_config["used_band"]
        self.tile_width = self.used_config["tile_width"]
        self.tile_height = self.used_config["tile_height"]
        self.threshold_disp_range_to_borders = self.used_config[
            "threshold_disp_range_to_borders"
        ]

    def get_required_bands(self):
        """
        Get bands required by this method.

        :return: required bands for left and right image
        :rtype: dict
        """
        return {
            "left": [self.used_band],
            "right": [self.used_band],
        }

    def check_conf(self, conf):
        """
        Merge user configuration with default values and validate schema.
        Extra keys in conf are preserved and ignored during schema validation.
        """

        if conf is None:
            conf = {}

        default_conf = {
            "conf_to_use": "default",
            "step": [10, 10],
            "loader_conf": None,
            "used_band": "b0",
            "threshold_disp_range_to_borders": False,
            "loader": "pandora2d",
            "tile_width": 500,
            "tile_height": 60,
        }

        loader_conf = conf.get("loader_conf", None)

        used_conf = default_conf.copy()
        used_conf.update(conf)
        method = used_conf.get("conf_to_use", "default")

        pandora_loader = Pandora2DLoader(
            conf=loader_conf,
            method_name=method,
            step=conf.get("step", [10, 10]),
        )

        # Get params from loaders
        self.loader = pandora_loader
        self.corr_config = collections.OrderedDict(pandora_loader.get_conf())

        used_conf["loader_conf"] = self.corr_config

        for elem in used_conf["step"]:
            if not isinstance(elem, int):
                raise RuntimeError("The step values should be integer")
            if elem < 0:
                raise RuntimeError("The step values should be positive")

        # Validate only keys defined in schema
        conf_to_check = {k: used_conf[k] for k in self.schema if k in used_conf}

        checker = Checker(self.schema)
        checker.validate(conf_to_check)

        return conf_to_check

    def add_margin_wrapper(self, margins_fun, method_margins):
        """
        Add pandora2d margins
        """

        def wrapped(row_min, row_max, col_min, col_max):
            """
            wrappers
            """
            margins_dataset = margins_fun(row_min, row_max, col_min, col_max)

            method_array = np.array(
                [
                    method_margins["left"],
                    method_margins["up"],
                    method_margins["right"],
                    method_margins["down"],
                ]
            )

            margins_dataset["left_margin"] += method_array
            margins_dataset["right_margin"] += method_array

            return margins_dataset

        return wrapped

    def crop_range(
        self,
        disp_min,
        disp_max,
        row_disp,
        nrows,
        ncols,
        max_ram_per_worker,
        pandora2d_machine,
    ):  # pylint: disable=too-many-positional-arguments
        """ "
        Crop the disparity range if it exceed the memory
        """
        conf_for_estimation = copy.deepcopy(self.corr_config)

        col_disparity = {
            "init": (disp_min + disp_max) / 2,
            "range": (disp_max - disp_min) / 2,
        }
        row_disparity = {
            "init": (row_disp[0] + row_disp[1]) / 2,
            "range": (row_disp[1] - row_disp[0]) / 2,
        }

        conf_for_estimation["input"]["col_disparity"] = col_disparity
        conf_for_estimation["input"]["row_disparity"] = row_disparity

        memory_mb = estimate_total_consumption(
            conf_for_estimation,
            nrows,
            ncols,
            pandora2d_machine.margins_disp.global_margins,
        )

        max_range = sm_wrap.get_max_disp_from_tile_memory(
            memory_mb, int(disp_max - disp_min), max_ram_per_worker
        )

        if int(disp_max - disp_min) > max_range:
            logging.warning("disparity range for current tile is cropped")

            disp_min = np.floor(disp_min * max_range / (disp_max - disp_min))
            disp_max = np.ceil(disp_max * max_range / (disp_max - disp_min))

        return disp_min, disp_max

    def run(
        self,
        left_image_object,
        right_image_object,
        saving_info_left=None,
        disp_lower_bound=None,
        disp_upper_bound=None,
        classif_bands_to_mask=None,
        disp_range_grid=None,
        row_disp=None,
        max_ram_per_worker=500,
    ):  # pylint: disable=R0917
        """
        Compute and filter sparse matches for one pair of epipolar tiles.
        """

        # transform disp_range_grid back to dict
        disp_range_grid = disp_range_grid.data

        if disp_range_grid is not None:
            # Generate disparity grids
            (
                disp_min_grid,
                disp_max_grid,
            ) = dm_algo.compute_disparity_grid(
                disp_range_grid,
                left_image_object,
                right_image_object,
                self.used_band,
                self.threshold_disp_range_to_borders,
            )

            disp_min = np.floor(np.min(disp_min_grid))
            disp_max = np.ceil(np.max(disp_max_grid))
        else:
            disp_min = np.floor(disp_lower_bound)
            disp_max = np.ceil(disp_upper_bound)

        # Load pandora plugin
        pandora2d.import_plugin()

        pandora2d_machine = Pandora2DMachine()

        # Put disparity in datasets
        nrows, ncols = left_image_object["im"].shape[1:]

        # For pandora2d, data arrays need to be 2d
        left_image_object["msk"] = xr.DataArray(
            data=left_image_object["msk"].values[0], dims=["row", "col"]
        )

        right_image_object["msk"] = xr.DataArray(
            data=right_image_object["msk"].values[0], dims=["row", "col"]
        )

        left_image_object["im"] = xr.DataArray(
            data=left_image_object["im"].values[0], dims=["row", "col"]
        )

        right_image_object["im"] = xr.DataArray(
            data=right_image_object["im"].values[0], dims=["row", "col"]
        )

        # Crop the disparity range if necessary
        disp_min, disp_max = self.crop_range(
            disp_min,
            disp_max,
            row_disp,
            nrows,
            ncols,
            max_ram_per_worker,
            pandora2d_machine,
        )

        # Define the disparity grid min max in the dataset
        left_disparity_col = xr.DataArray(
            data=np.stack(
                [
                    np.full((nrows, ncols), disp_min),
                    np.full((nrows, ncols), disp_max),
                ]
            ),
            dims=["band_disp", "row", "col"],
            coords={"band_disp": ["min", "max"]},
        )

        left_disparity_row = xr.DataArray(
            data=np.stack(
                [
                    np.full((nrows, ncols), row_disp[0]),
                    np.full((nrows, ncols), row_disp[1]),
                ]
            ),
            dims=["band_disp", "row", "col"],
            coords={"band_disp": ["min", "max"]},
        )

        left_disparity_col.attrs["no_data"] = self.corr_config["input"][
            "nodata_left"
        ]

        left_disparity_row.attrs["no_data"] = self.corr_config["input"][
            "nodata_left"
        ]

        left_image_object["col_disparity"] = left_disparity_col
        left_image_object["row_disparity"] = left_disparity_row
        check_disparities_from_dataset(left_image_object["col_disparity"])
        check_disparities_from_dataset(left_image_object["row_disparity"])

        left_image_object.attrs["col_disparity_source"] = [disp_min, disp_max]
        left_image_object.attrs["row_disparity_source"] = [disp_min, disp_max]

        left_image_object.attrs[cst.EPI_NO_DATA_IMG] = self.corr_config[
            "input"
        ]["nodata_left"]
        right_image_object.attrs[cst.EPI_NO_DATA_IMG] = self.corr_config[
            "input"
        ]["nodata_right"]

        # read images
        check_datasets(
            left_image_object,
            right_image_object,
        )

        # trigger all the steps of the machine at ones
        ref, _ = pandora2d.run(
            pandora2d_machine,
            left_image_object,
            right_image_object,
            self.corr_config,
        )

        # Compute matches
        row_map = ref["row_map"].values
        col_map = ref["col_map"].values

        rows = np.arange(
            left_image_object.roi_with_margins[1],
            left_image_object.roi_with_margins[3],
            step=self.step[0],
        )
        cols = np.arange(
            left_image_object.roi_with_margins[0],
            left_image_object.roi_with_margins[2],
            step=self.step[1],
        )

        cols_mesh, rows_mesh = np.meshgrid(cols, rows)
        left_points = np.column_stack(
            (cols_mesh.ravel(), rows_mesh.ravel())
        ).astype(float)

        right_points = np.copy(left_points)

        right_points[:, 0] += col_map.ravel()
        right_points[:, 1] += row_map.ravel()

        matches = np.column_stack((left_points, right_points))

        matches = matches[~np.isnan(matches).any(axis=1)]

        left_matches_dataframe = pandas.DataFrame(matches)

        cars_dataset.fill_dataframe(
            left_matches_dataframe,
            saving_info=saving_info_left,
            attributes=None,
        )

        return left_matches_dataframe
