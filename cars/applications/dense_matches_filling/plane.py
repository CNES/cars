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


import copy

# Standard imports
import logging

# Third party imports
from json_checker import Checker, Or
from shapely.geometry import Polygon

# CARS imports
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.dense_matches_filling import (
    fill_disp_constants as fd_cst,
)
from cars.applications.dense_matches_filling import fill_disp_tools as fd_tools
from cars.applications.dense_matches_filling.dense_matches_filling import (
    DenseMatchesFilling,
)
from cars.core import constants as cst
from cars.data_structures import cars_dataset, corresponding_tiles_tools


class PlaneFill(
    DenseMatchesFilling, short_name=["plane"]
):  # pylint: disable=R0903
    """
    Fill invalid area in disparity map using plane method
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf=None):
        """
        Init function of FillDisp

        :param conf: configuration for filling
        :return: a application_to_use object
        """

        super().__init__(conf=conf)

        # get conf
        self.used_method = self.used_config["method"]
        self.interpolation_type = self.used_config[fd_cst.INTERP_TYPE]
        self.interpolation_method = self.used_config[fd_cst.INTERP_METHOD]
        self.max_search_distance = self.used_config[fd_cst.MAX_DIST]
        self.smoothing_iterations = self.used_config[fd_cst.SMOOTH_IT]
        self.ignore_nodata_at_disp_mask_borders = self.used_config[
            fd_cst.IGNORE_NODATA
        ]
        self.ignore_zero_fill_disp_mask_values = self.used_config[
            fd_cst.IGNORE_ZERO
        ]
        self.ignore_extrema_disp_values = self.used_config[
            fd_cst.IGNORE_EXTREMA
        ]
        self.nb_pix = self.used_config["nb_pix"]
        self.percent_to_erode = self.used_config["percent_to_erode"]
        self.classification = self.used_config["classification"]
        # Saving files
        self.save_disparity_map = self.used_config["save_disparity_map"]

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
            fd_cst.IGNORE_NODATA, False
        )
        overloaded_conf[fd_cst.IGNORE_ZERO] = conf.get(fd_cst.IGNORE_ZERO, True)
        overloaded_conf[fd_cst.IGNORE_EXTREMA] = conf.get(
            fd_cst.IGNORE_EXTREMA, True
        )
        overloaded_conf["nb_pix"] = conf.get("nb_pix", 20)
        overloaded_conf["percent_to_erode"] = conf.get("percent_to_erode", 0.2)
        overloaded_conf["classification"] = conf.get("classification", None)
        # Saving files
        overloaded_conf["save_disparity_map"] = conf.get(
            "save_disparity_map", False
        )

        application_schema = {
            "method": str,
            "save_disparity_map": bool,
            "interpolation_type": Or(None, str),
            "interpolation_method": Or(None, str),
            "max_search_distance": Or(None, int),
            "smoothing_iterations": Or(None, int),
            "ignore_nodata_at_disp_mask_borders": bool,
            "ignore_zero_fill_disp_mask_values": bool,
            "ignore_extrema_disp_values": bool,
            "nb_pix": Or(None, int),
            "percent_to_erode": Or(None, float),
            "classification": Or(None, [str]),
        }

        # Check conf
        checker = Checker(application_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def get_poly_margin(self):
        """
        Get the margin used for polygon

        :return: self.nb_pix
        :rtype: int
        """

        return self.nb_pix

    def run(
        self,
        epipolar_disparity_map,
        holes_bbox_left=None,
        holes_bbox_right=None,
        disp_min=0,
        disp_max=0,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
    ):
        """
        Run Refill application using plane method.

        :param epipolar_disparity_map:  left to right disparity
        :type epipolar_disparity_map: CarsDataset
        :param holes_bbox_left:  left holes
        :type holes_bbox_left: CarsDataset
        :param holes_bbox_right:  right holes
        :type holes_bbox_right: CarsDataset
        :param disp_min: minimum disparity
        :type disp_min: int
        :param disp_max: maximum disparity
        :type disp_max: int
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param pair_key: pair id
        :type pair_key: str

        :return: filled disparity map: \
            Each CarsDataset contains:

            - N x M Delayed tiles.\
              Each tile will be a future xarray Dataset containing:
                - data with keys : "disp", "disp_msk"
                - attrs with keys: profile, window, overlaps
            - attributes containing:
                "largest_epipolar_region","opt_epipolar_tile_size",
                    "epipolar_regions_grid"

        :rtype: CarsDataset

        """
        if holes_bbox_left is None or holes_bbox_right is None:
            raise RuntimeError("Disparity holes bbox are inconsistent.")

        res = None

        if not self.classification:
            logging.info("Disparity holes filling was not activated")
            res = epipolar_disparity_map

        else:
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

            if epipolar_disparity_map.dataset_type == "arrays":
                new_epipolar_disparity_map = self.__register_dataset__(
                    epipolar_disparity_map,
                    self.save_disparity_map,
                    pair_folder,
                    pair_key,
                    app_name="plane",
                )

                # Get saving infos in order to save tiles when they are computed
                [saving_info] = self.orchestrator.get_saving_infos(
                    [new_epipolar_disparity_map]
                )

                # Add infos to orchestrator.out_json
                updating_dict = {
                    application_constants.APPLICATION_TAG: {
                        pair_key: {
                            fd_cst.FILL_DISP_WITH_PLAN_RUN_TAG: {},
                        }
                    }
                }
                self.orchestrator.update_out_info(updating_dict)
                logging.info(
                    "Fill missing disparity with plan model"
                    ": number tiles: {}".format(
                        epipolar_disparity_map.shape[1]
                        * epipolar_disparity_map.shape[0]
                    )
                )

                # get all polygones
                poly_list_left = fd_tools.get_polygons_from_cars_ds(
                    holes_bbox_left
                )
                poly_list_right = fd_tools.get_polygons_from_cars_ds(
                    holes_bbox_right
                )

                # Estimate right poly on left
                # using disparity range
                poly_list_right_on_left = [
                    fd_tools.estimate_poly_with_disp(
                        p, dmin=-disp_max, dmax=-disp_min
                    )
                    for p in poly_list_right
                ]

                # Merge polygones
                merged_poly_list = fd_tools.merge_intersecting_polygones(
                    poly_list_left + poly_list_right_on_left
                )

                logging.info(
                    "Disparity filling: {} holes on"
                    " left to fill".format(len(merged_poly_list))
                )

                # Generate polygones for tiles
                tiles_polygones = {}
                for col in range(epipolar_disparity_map.shape[1]):
                    for row in range(epipolar_disparity_map.shape[0]):
                        tile = epipolar_disparity_map.tiling_grid[row, col]
                        tiles_polygones[(row, col)] = Polygon(
                            [
                                [tile[0], tile[2]],
                                [tile[0], tile[3]],
                                [tile[1], tile[3]],
                                [tile[1], tile[2]],
                                [tile[0], tile[2]],
                            ]
                        )

                # Generate disparity maps
                for col in range(epipolar_disparity_map.shape[1]):
                    for row in range(epipolar_disparity_map.shape[0]):
                        if epipolar_disparity_map[row, col] is not None:
                            tile_poly = tiles_polygones[(row, col)]
                            # Get intersecting holes poly
                            corresponding_holes = (
                                fd_tools.get_corresponding_holes(
                                    tile_poly, merged_poly_list
                                )
                            )

                            # Get corresponding_tiles
                            # list of (tile_window, tile overlap, xr.Dataset)
                            corresponding_tiles = (
                                fd_tools.get_corresponding_tiles(
                                    tiles_polygones,
                                    corresponding_holes,
                                    epipolar_disparity_map,
                                )
                            )

                            # get tile window and overlap
                            window = new_epipolar_disparity_map.tiling_grid[
                                row, col
                            ]
                            overlap = new_epipolar_disparity_map.overlaps[
                                row, col
                            ]

                            # update saving infos  for potential replacement
                            full_saving_info = ocht.update_saving_infos(
                                saving_info, row=row, col=col
                            )

                            if len(corresponding_tiles) == 0:
                                # copy dataset
                                (
                                    new_epipolar_disparity_map[row, col]
                                ) = self.orchestrator.cluster.create_task(
                                    wrapper_copy_disparity
                                )(
                                    epipolar_disparity_map[row, col],
                                    window,
                                    overlap,
                                    saving_info=full_saving_info,
                                )

                            else:
                                # Fill holes
                                (
                                    new_epipolar_disparity_map[row, col]
                                ) = self.orchestrator.cluster.create_task(
                                    fill_disparity_plane_wrapper
                                )(
                                    corresponding_tiles,
                                    corresponding_holes,
                                    window,
                                    overlap,
                                    self.classification,
                                    ignore_nodata_at_disp_mask_borders=(
                                        self.ignore_nodata_at_disp_mask_borders
                                    ),
                                    ignore_zero_fill_disp_mask_values=(
                                        self.ignore_zero_fill_disp_mask_values
                                    ),
                                    ignore_extrema_disp_values=(
                                        self.ignore_extrema_disp_values
                                    ),
                                    nb_pix=self.nb_pix,
                                    percent_to_erode=self.percent_to_erode,
                                    interp_options=interp_options,
                                    saving_info=full_saving_info,
                                )

                res = new_epipolar_disparity_map

            else:
                logging.error(
                    "FillDisp application doesn't support "
                    "this input data format"
                )
        return res


def fill_disparity_plane_wrapper(
    corresponding_tiles,
    corresponding_poly,
    window,
    overlap,
    classification,
    ignore_nodata_at_disp_mask_borders=True,
    ignore_zero_fill_disp_mask_values=True,
    ignore_extrema_disp_values=True,
    nb_pix=20,
    percent_to_erode=0.3,
    interp_options=None,
    saving_info=None,
):
    """
    Wrapper to Fill disparity map holes

    :param corresponding_tiles: disparity map tiles
    :type corresponding_tiles: list(tuple(list, list, xr.Dataset))
    :param corresponding_poly: holes polygons
    :type corresponding_poly: list(Polygon)
    :param window: window of base tile [row min, row max, col min col max]
    :type window: list
    :param overlap: overlap [row min, row max, col min col max]
    :type overlap: list
    :param left_epi_image: left epipolar image
    :type left_epi_image:  xr.Dataset
    :param classification: list of tag to use
    :type classification: list(str)
    :param ignore_nodata_at_disp_mask_borders: ingore nodata
    :type ignore_nodata_at_disp_mask_borders: bool
    :param ignore_zero_fill_disp_mask_values: ingnore zero fill
    :type ignore_zero_fill_disp_mask_values: bool
    :param ignore_extrema_disp_values: ignore extrema
    :type ignore_extrema_disp_values: bool
    :param nb_pix: margin to use
    :type nb_pix: int
    :param percent_to_erode: percent to erode
    :type percent_to_erode: float
    :param interp_options: interp_options
    :type interp_options: dict
    :param saving_info: saving infos
    :type saving_info: dict


    :return: disp map
    :rtype: xr.Dataset
    """

    # Create combined xarray Dataset
    (
        combined_dataset,
        row_min,
        col_min,
    ) = corresponding_tiles_tools.reconstruct_data(
        corresponding_tiles, window, overlap
    )

    # Add a band to disparity dataset to memorize which pixels are filled
    combined_dataset = fd_tools.add_empty_filling_band(
        combined_dataset, ["plane.hole_center", "plane.hole_border"]
    )

    # Fill disparity
    fd_tools.fill_disp_using_plane(
        combined_dataset,
        corresponding_poly,
        row_min,
        col_min,
        ignore_nodata_at_disp_mask_borders,
        ignore_zero_fill_disp_mask_values,
        ignore_extrema_disp_values,
        nb_pix,
        percent_to_erode,
        interp_options,
        classification,
    )

    # Find xarray Dataset corresponding to current tile
    input_disp = copy.deepcopy(
        corresponding_tiles_tools.find_tile_dataset(corresponding_tiles, window)
    )

    # Add additional attribute "filling" to dataset template
    input_disp[cst.EPI_FILLING] = combined_dataset[cst.EPI_FILLING]

    # Crop Dataset to get tile disparity
    cropped_disp = corresponding_tiles_tools.crop_dataset(
        combined_dataset,
        input_disp,
        window,
        overlap,
        row_min,
        col_min,
    )

    # Fill with attributes
    cars_dataset.fill_dataset(
        cropped_disp,
        saving_info=saving_info,
        window=cars_dataset.window_array_to_dict(window),
        profile=None,
        attributes=None,
        overlaps=cars_dataset.overlap_array_to_dict(overlap),
    )

    return cropped_disp


def wrapper_copy_disparity(
    disp,
    window,
    overlap,
    saving_info=None,
):
    """
    Wrapper to copy previous disparity with additional filling band with zeros

    :param disp: disparity map
    :type disp: xr.Dataset
    :param window: window of base tile [row min, row max, col min col max]
    :type window: list
    :param overlap: overlap [row min, row max, col min col max]
    :type overlap: list
    :param saving_info: saving infos
    :type saving_info: dict

    :return: disp map
    :rtype: xr.Dataset
    """
    # Fill band named filling with zeros
    res = fd_tools.add_empty_filling_band(
        disp, ["plane.hole_center", "plane.hole_border"]
    )

    # Fill with attributes
    cars_dataset.fill_dataset(
        res,
        saving_info=saving_info,
        window=cars_dataset.window_array_to_dict(window),
        profile=None,
        attributes=None,
        overlaps=cars_dataset.overlap_array_to_dict(overlap),
    )

    return res
