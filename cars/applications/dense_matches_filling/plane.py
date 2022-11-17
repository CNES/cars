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
import copy
import logging
import os

# Third party imports
import numpy as np
from json_checker import Checker, Or
from shapely.geometry import Polygon

# CARS imports
import cars.orchestrator.orchestrator as ocht
from cars.applications.dense_matches_filling import (
    fill_disp_constants as fd_cst,
)
from cars.applications.dense_matches_filling import fill_disp_tools as fd_tools
from cars.applications.dense_matches_filling.dense_matches_filling import (
    DenseMatchingFiling,
)
from cars.applications.dense_matching import dense_matching_tools
from cars.core import constants as cst
from cars.core import constants_disparity as cst_disp
from cars.data_structures import cars_dataset, corresponding_tiles_tools


class PlaneFill(
    DenseMatchingFiling, short_name=["plane"]
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

        # Check conf
        checked_conf = self.check_conf(conf)
        # used_config used for printing config
        self.used_config = checked_conf

        # check conf
        self.activated = checked_conf["activated"]
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

        # Saving files
        self.save_disparity_map = checked_conf["save_disparity_map"]

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

        overloaded_conf["activated"] = conf.get("activated", False)

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

        # Saving files
        overloaded_conf["save_disparity_map"] = conf.get(
            "save_disparity_map", False
        )

        application_schema = {
            "method": str,
            "save_disparity_map": bool,
            "activated": bool,
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

    def get_is_activated(self):
        """
        Get the activated attribute

        :return: self.activated
        :rtype: bool
        """

        return self.activated

    def get_poly_margin(self):
        """
        Get the margin used for polygon

        :return: self.nb_pix
        :rtype: int
        """

        return self.nb_pix

    def run(
        self,
        epipolar_disparity_map_left,
        epipolar_disparity_map_right,
        epipolar_images_left,
        holes_bbox_left,
        holes_bbox_right,
        disp_min=0,
        disp_max=0,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
    ):
        """
        Run Refill application using plane method.

        :param epipolar_disparity_map_left:  left disparity
        :type epipolar_disparity_map_left: CarsDataset
        :param epipolar_disparity_map_right:  right disparity
        :type epipolar_disparity_map_right: CarsDataset
        :param epipolar_images_left: tiled left epipolar CarsDataset contains:

                - N x M Delayed tiles. \
                    Each tile will be a future xarray Dataset containing:

                    - data with keys : "im", "msk", "color"
                    - attrs with keys: "margins" with "disp_min" and "disp_max"\
                        "transform", "crs", "valid_pixels", "no_data_mask",\
                        "no_data_img"
                - attributes containing:
                    "largest_epipolar_region","opt_epipolar_tile_size",
                    "epipolar_regions_grid"
        :type epipolar_images_left: CarsDataset
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

        :return: filled left disparity map, filled right disparity map: \
            Each CarsDataset contains:

            - N x M Delayed tiles.\
              Each tile will be a future xarray Dataset containing:
                - data with keys : "disp", "disp_msk"
                - attrs with keys: profile, window, overlaps
            - attributes containing:
                "largest_epipolar_region","opt_epipolar_tile_size",
                    "epipolar_regions_grid"

        :rtype: Tuple(CarsDataset, CarsDataset)

        """

        res = None, None

        if not self.activated:
            logging.info("Disparity hiles filing was not activated")
            res = epipolar_disparity_map_left, epipolar_disparity_map_right

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

            if epipolar_disparity_map_left.dataset_type == "arrays":
                # Create CarsDataset
                # Epipolar_disparity
                new_epipolar_disparity_map_left = cars_dataset.CarsDataset(
                    "arrays"
                )
                new_epipolar_disparity_map_left.create_empty_copy(
                    epipolar_disparity_map_left
                )

                new_epipolar_disparity_map_right = cars_dataset.CarsDataset(
                    "arrays"
                )
                new_epipolar_disparity_map_right.create_empty_copy(
                    epipolar_disparity_map_right
                )

                # Update attributes to get epipolar info
                new_epipolar_disparity_map_left.attributes.update(
                    epipolar_disparity_map_left.attributes
                )

                # Save disparity maps
                if self.save_disparity_map:
                    self.orchestrator.add_to_save_lists(
                        os.path.join(pair_folder, "epi_disp_filled_left.tif"),
                        cst_disp.MAP,
                        new_epipolar_disparity_map_left,
                        cars_ds_name="epi_disp_filled_left",
                    )

                    self.orchestrator.add_to_save_lists(
                        os.path.join(pair_folder, "epi_disp_filled_right.tif"),
                        cst_disp.MAP,
                        new_epipolar_disparity_map_right,
                        cars_ds_name="epi_disp_filled_right",
                    )

                    self.orchestrator.add_to_save_lists(
                        os.path.join(
                            pair_folder, "epi_disp_color_filled_left.tif"
                        ),
                        cst.EPI_COLOR,
                        new_epipolar_disparity_map_left,
                        cars_ds_name="epi_disp_color_filled_left",
                    )

                    self.orchestrator.add_to_save_lists(
                        os.path.join(
                            pair_folder, "epi_disp_color_filled_right.tif"
                        ),
                        cst.EPI_COLOR,
                        new_epipolar_disparity_map_right,
                        cars_ds_name="epi_disp_color_filled_right",
                    )

                    self.orchestrator.add_to_save_lists(
                        os.path.join(
                            pair_folder, "epi_disp_mask_filled_left.tif"
                        ),
                        cst_disp.VALID,
                        new_epipolar_disparity_map_left,
                        cars_ds_name="epi_disp_mask_filled_left",
                    )

                    self.orchestrator.add_to_save_lists(
                        os.path.join(
                            pair_folder, "epi_disp_mask_filled_right.tif"
                        ),
                        cst_disp.VALID,
                        new_epipolar_disparity_map_right,
                        cars_ds_name="epi_disp_mask_filled_right",
                    )

                    self.orchestrator.add_to_save_lists(
                        os.path.join(
                            pair_folder, "epi_ambiguity_filled_left.tif"
                        ),
                        cst_disp.AMBIGUITY_CONFIDENCE,
                        new_epipolar_disparity_map_left,
                        cars_ds_name="epi_ambiguity_filled_left",
                    )

                    self.orchestrator.add_to_save_lists(
                        os.path.join(
                            pair_folder, "epi_ambiguity_filled_right.tif"
                        ),
                        cst_disp.AMBIGUITY_CONFIDENCE,
                        new_epipolar_disparity_map_right,
                        cars_ds_name="epi_ambiguity_filled_right",
                    )

                # Get saving infos in order to save tiles when they are computed
                [
                    saving_info_left,
                    saving_info_right,
                ] = self.orchestrator.get_saving_infos(
                    [
                        new_epipolar_disparity_map_left,
                        new_epipolar_disparity_map_right,
                    ]
                )

                # get all polygones
                poly_list_left = fd_tools.get_polygons_from_cars_ds(
                    holes_bbox_left
                )
                poly_list_right = fd_tools.get_polygons_from_cars_ds(
                    holes_bbox_right
                )

                # Estimate right poly on left and left poly on right
                # using disparity range
                poly_list_right_on_left = [
                    fd_tools.estimate_poly_with_disp(
                        p, dmin=-disp_max, dmax=-disp_min
                    )
                    for p in poly_list_right
                ]
                poly_list_left_on_right = [
                    fd_tools.estimate_poly_with_disp(
                        p, dmin=disp_min, dmax=disp_max
                    )
                    for p in poly_list_left
                ]

                # Merge polygones
                merged_poly_list_left = fd_tools.merge_intersecting_polygones(
                    poly_list_left + poly_list_right_on_left
                )
                merged_poly_list_right = fd_tools.merge_intersecting_polygones(
                    poly_list_right + poly_list_left_on_right
                )

                logging.info(
                    "Disparity filing: {} holes on"
                    " left to fill, {} on right".format(
                        len(merged_poly_list_left), len(merged_poly_list_right)
                    )
                )

                # Generate polygones for tiles
                tiles_polygones = {}
                for col in range(epipolar_disparity_map_right.shape[1]):
                    for row in range(epipolar_disparity_map_right.shape[0]):
                        tile = epipolar_disparity_map_right.tiling_grid[
                            row, col
                        ]
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
                for col in range(epipolar_disparity_map_right.shape[1]):
                    for row in range(epipolar_disparity_map_right.shape[0]):

                        tile_poly = tiles_polygones[(row, col)]
                        # Get intersecting holes poly
                        corresponding_holes_left = (
                            fd_tools.get_corresponding_holes(
                                tile_poly, merged_poly_list_left
                            )
                        )
                        corresponding_holes_right = (
                            fd_tools.get_corresponding_holes(
                                tile_poly, merged_poly_list_right
                            )
                        )
                        # Get corresponding_tiles
                        # list of (tile_window, tile overlap, xr.Dataset)
                        corresponding_tiles_left = (
                            fd_tools.get_corresponding_tiles(
                                tiles_polygones,
                                corresponding_holes_left,
                                epipolar_disparity_map_left,
                            )
                        )
                        corresponding_tiles_right = (
                            fd_tools.get_corresponding_tiles(
                                tiles_polygones,
                                corresponding_holes_right,
                                epipolar_disparity_map_right,
                            )
                        )

                        # get tile window and overlap
                        window = new_epipolar_disparity_map_left.tiling_grid[
                            row, col
                        ]
                        overlap_left = new_epipolar_disparity_map_left.overlaps[
                            row, col
                        ]
                        overlap_right = (
                            new_epipolar_disparity_map_right.overlaps[row, col]
                        )

                        if (
                            len(corresponding_tiles_left)
                            + len(corresponding_tiles_right)
                            == 0
                        ):
                            # copy dataset
                            (
                                new_epipolar_disparity_map_left[row, col],
                                new_epipolar_disparity_map_right[row, col],
                            ) = self.orchestrator.cluster.create_task(
                                wrapper_copy_disparity, nout=2
                            )(
                                epipolar_disparity_map_left[row, col],
                                epipolar_disparity_map_right[row, col],
                                window,
                                overlap_left,
                                overlap_right,
                                saving_info_left=saving_info_left,
                                saving_info_right=saving_info_right,
                            )

                        else:
                            # Fill holes
                            (
                                new_epipolar_disparity_map_left[row, col],
                                new_epipolar_disparity_map_right[row, col],
                            ) = self.orchestrator.cluster.create_task(
                                wrapper_fill_disparity, nout=2
                            )(
                                corresponding_tiles_left,
                                corresponding_tiles_right,
                                corresponding_holes_left,
                                corresponding_holes_right,
                                window,
                                overlap_left,
                                overlap_right,
                                epipolar_images_left[row, col],
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
                                saving_info_left=saving_info_left,
                                saving_info_right=saving_info_right,
                            )
                res = (
                    new_epipolar_disparity_map_left,
                    new_epipolar_disparity_map_right,
                )

            else:
                logging.error(
                    "FillDisp application doesn't support "
                    "this input data format"
                )
        return res


def wrapper_fill_disparity(
    corresponding_tiles_left,
    corresponding_tiles_right,
    corresponding_poly_left,
    corresponding_poly_right,
    window,
    overlap_left,
    overlap_right,
    left_epi_image,
    ignore_nodata_at_disp_mask_borders=True,
    ignore_zero_fill_disp_mask_values=True,
    ignore_extrema_disp_values=True,
    nb_pix=20,
    percent_to_erode=0.3,
    interp_options=None,
    saving_info_left=None,
    saving_info_right=None,
):
    """
    Wrapper to Fill disparity map holes

    :param corresponding_tiles_left: left disparity map tiles
    :type corresponding_tiles_left: list(tuple(list, list, xr.Dataset))
    :param corresponding_tiles_right: right disparity map tiles
    :type corresponding_tiles_right: list(tuple(list, list, xr.Dataset))
    :param corresponding_poly_left: left holes polygons
    :type corresponding_poly_left: list(Polygon)
    :param corresponding_poly_right: right holes polygons
    :type corresponding_poly_right: list(Polygon)
    :param window: window of base tile [row min, row max, col min col max]
    :type window: list
    :param overlap_left: left overlap [row min, row max, col min col max]
    :type overlap_left: list
    :param overlap_right: left overlap  [row min, row max, col min col max]
    :type overlap_right: list
    :param left_epi_image: left epipolar image
    :type left_epi_image:  xr.Dataset
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
    :param saving_info_left: saving infos left
    :type saving_info_left: dict
    :param saving_info_right: saving infos right
    :type saving_info_right: dict

    :return: left disp map, right disp map
    :rtype: xr.Dataset, xr.Dataset
    """

    # find xarray Dataset corresponding to current tile
    input_disp_left = corresponding_tiles_tools.find_tile_dataset(
        corresponding_tiles_left, window
    )

    # Create combined xarray Dataset

    (
        combined_dataset_left,
        row_min_left,
        col_min_left,
    ) = corresponding_tiles_tools.reconstruct_data(
        corresponding_tiles_left, window, overlap_left
    )

    # Fill disparity

    fd_tools.fill_disp_using_plane(
        combined_dataset_left,
        corresponding_poly_left,
        row_min_left,
        col_min_left,
        ignore_nodata_at_disp_mask_borders,
        ignore_zero_fill_disp_mask_values,
        ignore_extrema_disp_values,
        nb_pix,
        percent_to_erode,
        interp_options,
    )
    # Crop Dataset to get tile disparity
    # crop
    croped_disp_left = corresponding_tiles_tools.crop_dataset(
        combined_dataset_left,
        input_disp_left,
        window,
        overlap_left,
        row_min_left,
        col_min_left,
    )

    # find xarray Dataset corresponding to current tile
    input_disp_right = corresponding_tiles_tools.find_tile_dataset(
        corresponding_tiles_right, window
    )

    croped_disp_right = None
    if input_disp_right is not None:

        (
            combined_dataset_right,
            row_min_right,
            col_min_right,
        ) = corresponding_tiles_tools.reconstruct_data(
            corresponding_tiles_right, window, overlap_right
        )

        fd_tools.fill_disp_using_plane(
            combined_dataset_right,
            corresponding_poly_right,
            row_min_right,
            col_min_right,
            ignore_nodata_at_disp_mask_borders,
            ignore_zero_fill_disp_mask_values,
            ignore_extrema_disp_values,
            nb_pix,
            percent_to_erode,
            interp_options,
        )

        # crop
        croped_disp_right = corresponding_tiles_tools.crop_dataset(
            combined_dataset_right,
            input_disp_right,
            window,
            overlap_right,
            row_min_right,
            col_min_right,
        )

        # compute right color image from right-left disparity map
        color_sec = dense_matching_tools.estimate_color_from_disparity(
            croped_disp_right,
            left_epi_image,
            croped_disp_left,
        )

        # check bands
        if len(left_epi_image[cst.EPI_COLOR].values.shape) > 2:
            nb_bands = left_epi_image[cst.EPI_COLOR].values.shape[0]
            if cst.BAND not in croped_disp_right.dims:
                croped_disp_right.assign_coords({cst.BAND: np.arange(nb_bands)})

        # merge colors
        croped_disp_right[cst.EPI_COLOR] = color_sec[cst.EPI_IMAGE]

    # Fill with attributes
    cars_dataset.fill_dataset(
        croped_disp_left,
        saving_info=saving_info_left,
        window=cars_dataset.window_array_to_dict(window),
        profile=None,
        attributes=None,
        overlaps=cars_dataset.overlap_array_to_dict(overlap_left),
    )

    if croped_disp_right is not None:
        cars_dataset.fill_dataset(
            croped_disp_right,
            saving_info=saving_info_right,
            window=cars_dataset.window_array_to_dict(window),
            profile=None,
            attributes=None,
            overlaps=cars_dataset.overlap_array_to_dict(overlap_right),
        )

    return croped_disp_left, croped_disp_right


def wrapper_copy_disparity(
    left_disp,
    right_disp,
    window,
    overlap_left,
    overlap_right,
    saving_info_left=None,
    saving_info_right=None,
):
    """
    Wrapper to copy previous disparity

    :param left_disp: left disparity map
    :type left_disp: xr.Dataset
    :param right_disp: right disparity map
    :type right_disp: xr.Dataset
    :param window: window of base tile [row min, row max, col min col max]
    :type window: list
    :param overlap_left: left overlap [row min, row max, col min col max]
    :type overlap_left: list
    :param overlap_right: left overlap  [row min, row max, col min col max]
    :type overlap_right: list
    :param saving_info_left: saving infos left
    :type saving_info_left: dict
    :param saving_info_right: saving infos right
    :type saving_info_right: dict

    :return: left disp map, right disp map
    :rtype: xr.Dataset, xr.Dataset
    """

    res = (copy.copy(left_disp), copy.copy(right_disp))

    # Fill with attributes
    cars_dataset.fill_dataset(
        res[0],
        saving_info=saving_info_left,
        window=cars_dataset.window_array_to_dict(window),
        profile=None,
        attributes=None,
        overlaps=cars_dataset.overlap_array_to_dict(overlap_left),
    )

    if res[1] is not None:
        cars_dataset.fill_dataset(
            res[1],
            saving_info=saving_info_right,
            window=cars_dataset.window_array_to_dict(window),
            profile=None,
            attributes=None,
            overlaps=cars_dataset.overlap_array_to_dict(overlap_right),
        )

    return res
