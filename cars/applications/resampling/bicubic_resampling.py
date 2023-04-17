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
this module contains the dense_matching application class.
"""
# pylint: disable=too-many-lines
# TODO refacto: factorize disributed code, and remove too-many-lines

# Standard imports
import logging
import os
from typing import Dict, Tuple

# Third party imports
import numpy as np
import xarray as xr
from json_checker import Checker
from shapely.geometry import Polygon

# CARS imports
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.grid_generation import grids
from cars.applications.resampling import resampling_constants, resampling_tools
from cars.applications.resampling.resampling import Resampling
from cars.core import constants as cst
from cars.core import inputs, tiling
from cars.core.datasets import get_color_bands
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset, format_transformation
from cars.pipelines.sensor_to_dense_dsm import (
    sensor_dense_dsm_constants as sens_cst,
)


class BicubicResampling(Resampling, short_name="bicubic"):
    """
    BicubicResampling
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf=None):
        """
        Init function of BicubicResampling

        :param conf: configuration for resampling
        :return: an application_to_use object
        """
        super().__init__(conf=conf)

        # check conf
        self.used_method = self.used_config["method"]
        self.epi_tile_size = self.used_config["epi_tile_size"]
        # Saving bools
        self.save_epipolar_image = self.used_config["save_epipolar_image"]
        self.save_epipolar_color = self.used_config["save_epipolar_color"]

        # check loader
        # TODO use loaders

        # Init orchestrator
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

        # get rasterization parameter
        overloaded_conf["method"] = conf.get("method", "bicubic")
        overloaded_conf["epi_tile_size"] = conf.get("epi_tile_size", 500)
        # Saving bools
        overloaded_conf["save_epipolar_image"] = conf.get(
            "save_epipolar_image", False
        )
        overloaded_conf["save_epipolar_color"] = conf.get(
            "save_epipolar_color", False
        )

        rectification_schema = {
            "method": str,
            "epi_tile_size": int,
            "save_epipolar_image": bool,
            "save_epipolar_color": bool,
        }

        # Check conf
        checker = Checker(rectification_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def pre_run(
        self,
        grid_left,
        optimum_tile_size,
    ):
        """
        Pre run some computations : tiling grid

        :param grid_left: left grid
        :type grid_left: CarsDataset
        :param optimum_tile_size: optimum tile size
        :type optimum_tile_size: int


        :return: epipolar_regions_grid, epipolar_regions,
            opt_epipolar_tile_size, largest_epipolar_region,
        """

        # Get largest epipolar regions from configuration file
        largest_epipolar_region = [
            0,
            0,
            grid_left.attributes["epipolar_size_x"],
            grid_left.attributes["epipolar_size_y"],
        ]

        origin = grid_left.attributes["grid_origin"]
        spacing = grid_left.attributes["grid_spacing"]

        logging.info(
            "Size of epipolar image: {}".format(largest_epipolar_region)
        )
        logging.debug("Origin of epipolar grid: {}".format(origin))
        logging.debug("Spacing of epipolar grid: {}".format(spacing))

        # get optimum tile_size
        if optimum_tile_size is None:
            opt_epipolar_tile_size = self.epi_tile_size
        else:
            opt_epipolar_tile_size = optimum_tile_size

        logging.info(
            "Optimal tile size for epipolar regions: "
            "{size}x{size} pixels".format(size=opt_epipolar_tile_size)
        )

        epipolar_regions_grid = tiling.generate_tiling_grid(
            0,
            0,
            grid_left.attributes["epipolar_size_y"],
            grid_left.attributes["epipolar_size_x"],
            opt_epipolar_tile_size,
            opt_epipolar_tile_size,
        )

        logging.info(
            "Epipolar image will be processed in {} splits".format(
                epipolar_regions_grid.shape[0] * epipolar_regions_grid.shape[1]
            )
        )

        return (
            epipolar_regions_grid,
            opt_epipolar_tile_size,
            largest_epipolar_region,
        )

    def run(
        self,
        sensor_image_left,
        sensor_image_right,
        grid_left,
        grid_right,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
        margins=None,
        optimum_tile_size=None,
        add_color=True,
        epipolar_roi=None,
    ):  # noqa: C901
        """
        Run resampling application.

        Creates left and right CarsDataset filled with xarray.Dataset,
        corresponding to sensor images resampled in epipolar geometry.

        :param sensor_images_left: tiled sensor left image
            Dict Must contain keys : "image", "color", "geomodel",
            "no_data", "mask", "classification". Paths must be absolutes
        :type sensor_images_left: CarsDataset
        :param sensor_images_right: tiled sensor right image
            Dict Must contain keys : "image", "color", "geomodel",
            "no_data", "mask", "classification". Paths must be absolutes
        :type sensor_images_right: CarsDataset
        :param grid_left: left epipolar grid
            Grid CarsDataset contains :

            - A single tile stored in [0,0], containing a (N, M, 2) shape
                array in xarray Dataset
            - Attributes containing: "grid_spacing", "grid_origin", \
                "epipolar_size_x", "epipolar_size_y", "epipolar_origin_x",\
                 "epipolar_origin_y", epipolar_spacing_x",\
                 "epipolar_spacing", "disp_to_alt_ratio",\
        :type grid_left: CarsDataset
        :param grid_right: right epipolar grid. Grid CarsDataset contains :

            - A single tile stored in [0,0], containing a (N, M, 2) shape \
                array in xarray Dataset
            - Attributes containing: "grid_spacing", "grid_origin",\
                "epipolar_size_x", "epipolar_size_y", "epipolar_origin_x",\
                 "epipolar_origin_y", epipolar_spacing_x",\
                 "epipolar_spacing", "disp_to_alt_ratio",
        :type grid_right: CarsDataset
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: directory to save files to
        :param pair_key: pair id
        :type pair_key: str
        :param margins: margins to use
        :type margins: xr.Dataset
        :param optimum_tile_size: optimum tile size to use
        :type optimum_tile_size: int
        :param add_color: add color image to dataset
        :type add_color: bool
        :param epipolar_roi: Epipolar roi to use if set.
            Set None tiles outsize roi
        :type epipolar_roi: list(int), [row_min, row_max,  col_min, col_max]

        :return: left epipolar image, right epipolar image. \
            Each CarsDataset contains:

            - N x M Delayed tiles. \
                Each tile will be a future xarray Dataset containing:

                - data with keys : "im", "msk", "color", "classif"
                - attrs with keys: "margins" with "disp_min" and "disp_max"\
                    "transform", "crs", "valid_pixels", "no_data_mask",
                    "no_data_img"
            - attributes containing: \
                "largest_epipolar_region","opt_epipolar_tile_size"

        :rtype: Tuple(CarsDataset, CarsDataset)
        """

        # Default orchestrator
        if orchestrator is None:
            # Create default sequential orchestrator for current application
            # be aware, no out_json will be shared between orchestrators
            # No files saved
            self.orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )
        else:
            self.orchestrator = orchestrator

        if pair_folder is None:
            pair_folder = os.path.join(self.orchestrator.out_dir, "tmp")
            safe_makedirs(pair_folder)

        # Create zeros margins if not provided
        if margins is None:
            corner = ["left", "up", "right", "down"]
            data = np.zeros(len(corner))
            col = np.arange(len(corner))
            margins = xr.Dataset(
                {"left_margin": (["col"], data)}, coords={"col": col}
            )
            margins["right_margin"] = xr.DataArray(data, dims=["col"])

        # Get grids and regions for current pair
        (
            epipolar_regions_grid,
            opt_epipolar_tile_size,
            largest_epipolar_region,
        ) = self.pre_run(
            grid_left,
            optimum_tile_size,
        )

        epipolar_images_left, epipolar_images_right = None, None

        # Retrieve number of bands
        if sens_cst.INPUT_COLOR in sensor_image_left:
            nb_bands = inputs.rasterio_get_nb_bands(
                sensor_image_left[sens_cst.INPUT_COLOR]
            )
        else:
            logging.info(
                "No color image has been given in input, "
                "{} will be used as the color image".format(
                    sensor_image_left[sens_cst.INPUT_IMG]
                )
            )

            nb_bands = inputs.rasterio_get_nb_bands(
                sensor_image_left[sens_cst.INPUT_IMG]
            )

        logging.info("Number of bands in color image: {}".format(nb_bands))

        # Create CarsDataset
        # Epipolar_images
        epipolar_images_left = cars_dataset.CarsDataset("arrays")
        epipolar_images_right = cars_dataset.CarsDataset("arrays")

        # Compute tiling grid
        epipolar_images_left.tiling_grid = epipolar_regions_grid

        # Generate tiling grid
        epipolar_images_right.tiling_grid = epipolar_regions_grid

        # Compute overlaps
        epipolar_images_left.overlaps = (
            format_transformation.grid_margins_2_overlaps(
                epipolar_images_left.tiling_grid, margins["left_margin"]
            )
        )
        epipolar_images_right.overlaps = (
            format_transformation.grid_margins_2_overlaps(
                epipolar_images_right.tiling_grid, margins["right_margin"]
            )
        )

        # update attributes
        epipolar_images_attributes = {
            "largest_epipolar_region": largest_epipolar_region,
            "opt_epipolar_tile_size": opt_epipolar_tile_size,
        }

        epipolar_images_left.attributes.update(epipolar_images_attributes)
        epipolar_images_right.attributes.update(epipolar_images_attributes)

        # Save objects
        if self.save_epipolar_image:
            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epi_img_left.tif"),
                cst.EPI_IMAGE,
                epipolar_images_left,
                cars_ds_name="epi_img_left",
            )
            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epi_img_right.tif"),
                cst.EPI_IMAGE,
                epipolar_images_right,
                cars_ds_name="epi_img_right",
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epi_img_left_mask.tif"),
                cst.EPI_MSK,
                epipolar_images_left,
                cars_ds_name="epi_img_left_mask",
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epi_img_right_mask.tif"),
                cst.EPI_MSK,
                epipolar_images_right,
                cars_ds_name="epi_img_right_mask",
            )

        if self.save_epipolar_color:
            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epi_color.tif"),
                cst.EPI_COLOR,
                epipolar_images_left,
                cars_ds_name="epi_color",
            )

        # Get saving infos in order to save tiles when they are computed
        [
            saving_info_left,
            saving_info_right,
        ] = self.orchestrator.get_saving_infos(
            [epipolar_images_left, epipolar_images_right]
        )

        logging.info(
            "Number of tiles in epipolar resampling :"
            "row : {} "
            "col : {}".format(
                epipolar_images_left.tiling_grid.shape[0],
                epipolar_images_left.tiling_grid.shape[1],
            )
        )

        # Add infos to orchestrator.out_json
        updating_dict = {
            application_constants.APPLICATION_TAG: {
                pair_key: {
                    resampling_constants.METHOD: self.used_method,
                    resampling_constants.RESAMPLING_RUN_TAG: {},
                }
            }
        }
        self.orchestrator.update_out_info(updating_dict)

        # Save grids on disk
        # TODO remove it
        # Save grids
        safe_makedirs(os.path.join(pair_folder, "tmp"))
        grid_origin = grid_left.attributes["grid_origin"]
        grid_spacing = grid_left.attributes["grid_spacing"]
        left_grid_path = grids.get_new_path(
            os.path.join(pair_folder, "tmp", "left_epi_grid.tif")
        )
        grids.write_grid(
            grid_left[0, 0], left_grid_path, grid_origin, grid_spacing
        )

        right_grid_path = grids.get_new_path(
            os.path.join(pair_folder, "tmp", "corrected_right_epi_grid.tif")
        )
        grids.write_grid(
            grid_right[0, 0], right_grid_path, grid_origin, grid_spacing
        )
        # End TODO

        # retrieves some data
        epipolar_size_x = grid_left.attributes["epipolar_size_x"]
        epipolar_size_y = grid_left.attributes["epipolar_size_y"]
        img1 = sensor_image_left[sens_cst.INPUT_IMG]
        img2 = sensor_image_right[sens_cst.INPUT_IMG]
        color1 = sensor_image_left.get(sens_cst.INPUT_COLOR, None)
        grid1 = left_grid_path
        grid2 = right_grid_path
        nodata1 = sensor_image_left.get(sens_cst.INPUT_NODATA, None)
        nodata2 = sensor_image_right.get(sens_cst.INPUT_NODATA, None)
        mask1 = sensor_image_left.get(sens_cst.INPUT_MSK, None)
        mask2 = sensor_image_right.get(sens_cst.INPUT_MSK, None)
        classif1 = sensor_image_left.get(sens_cst.INPUT_CLASSIFICATION, None)
        classif2 = sensor_image_right.get(sens_cst.INPUT_CLASSIFICATION, None)

        # Set Epipolar roi
        epi_tilling_grid = epipolar_images_left.tiling_grid
        if epipolar_roi is None:
            epipolar_roi = [
                np.min(epi_tilling_grid[:, :, 0]),
                np.max(epi_tilling_grid[:, :, 1]),
                np.min(epi_tilling_grid[:, :, 2]),
                np.max(epi_tilling_grid[:, :, 3]),
            ]
        # Convert roi to polygon
        epipolar_roi_poly = Polygon(
            [
                [epipolar_roi[0], epipolar_roi[2]],
                [epipolar_roi[0], epipolar_roi[3]],
                [epipolar_roi[1], epipolar_roi[3]],
                [epipolar_roi[1], epipolar_roi[2]],
                [epipolar_roi[0], epipolar_roi[2]],
            ]
        )

        # Generate Image pair
        for col in range(epipolar_images_left.shape[1]):
            for row in range(epipolar_images_left.shape[0]):
                # Create polygon corresponding to tile
                tile = epi_tilling_grid[row, col]
                tile_roi_poly = Polygon(
                    [
                        [tile[0], tile[2]],
                        [tile[0], tile[3]],
                        [tile[1], tile[3]],
                        [tile[1], tile[2]],
                        [tile[0], tile[2]],
                    ]
                )

                if epipolar_roi_poly.intersects(tile_roi_poly):
                    # get overlaps
                    left_overlap = cars_dataset.overlap_array_to_dict(
                        epipolar_images_left.overlaps[row, col]
                    )
                    right_overlap = cars_dataset.overlap_array_to_dict(
                        epipolar_images_right.overlaps[row, col]
                    )
                    # get window
                    left_window = epipolar_images_left.get_window_as_dict(
                        row, col
                    )

                    # update saving infos  for potential replacement
                    full_saving_info_left = ocht.update_saving_infos(
                        saving_info_left, row=row, col=col
                    )
                    full_saving_info_right = ocht.update_saving_infos(
                        saving_info_right, row=row, col=col
                    )

                    # Compute images
                    (
                        epipolar_images_left[row, col],
                        epipolar_images_right[row, col],
                    ) = self.orchestrator.cluster.create_task(
                        generate_epipolar_images_wrapper, nout=2
                    )(
                        left_overlap,
                        right_overlap,
                        left_window,
                        margins,
                        epipolar_size_x,
                        epipolar_size_y,
                        img1,
                        img2,
                        grid1,
                        grid2,
                        add_color=add_color,
                        color1=color1,
                        mask1=mask1,
                        mask2=mask2,
                        classif1=classif1,
                        classif2=classif2,
                        nodata1=nodata1,
                        nodata2=nodata2,
                        saving_info_left=full_saving_info_left,
                        saving_info_right=full_saving_info_right,
                    )
        return epipolar_images_left, epipolar_images_right


def generate_epipolar_images_wrapper(
    left_overlaps,
    right_overlaps,
    window,
    initial_margins,
    epipolar_size_x,
    epipolar_size_y,
    img1,
    img2,
    grid1,
    grid2,
    add_color=True,
    color1=None,
    mask1=None,
    mask2=None,
    classif1=None,
    classif2=None,
    nodata1=0,
    nodata2=0,
    saving_info_left=None,
    saving_info_right=None,
) -> Dict[str, Tuple[xr.Dataset, xr.Dataset]]:
    """
    Compute disparity maps from image objects. This function will be run
    as a delayed task. If user want to correctly save dataset, the user must
    provide saving_info_left and right.  See cars_dataset.fill_dataset.


    :param left_overlaps: Overlaps of left image, with row_min, row_max,
            col_min and col_max keys.
    :type left_overlaps: dict
    :param right_overlaps: Overlaps of right image, with row_min, row_max,
            col_min and col_max keys.
    :type right_overlaps: dict
    :param window: Window considered in generation, with row_min, row_max,
            col_min and col_max keys.
    :type window: dict
    :param initial_margins: Initial margins without crops (used as template)
    :type initial_margins: dict

    :return: Left image object, Right image object (if exists)

    Returned objects are composed of dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_COLOR (for left, if given)
    """
    region, margins = format_transformation.region_margins_from_window(
        initial_margins, window, left_overlaps, right_overlaps
    )

    # Rectify images
    (
        left_dataset,
        right_dataset,
        left_color_dataset,
        left_classif_dataset,
        right_classif_dataset,
    ) = resampling_tools.epipolar_rectify_images(
        img1,
        img2,
        grid1,
        grid2,
        region,
        margins,
        epipolar_size_x,
        epipolar_size_y,
        color1=color1,
        mask1=mask1,
        mask2=mask2,
        classif1=classif1,
        classif2=classif2,
        nodata1=nodata1,
        nodata2=nodata2,
        add_color=add_color,
    )

    if add_color:
        # merge color in left dataset
        if len(left_color_dataset[cst.EPI_IMAGE].values.shape) > 2:
            band_im = get_color_bands(left_color_dataset, cst.EPI_IMAGE)
        else:
            band_im = ["Gray"]

        # Add epi color mask if exists
        if cst.EPI_MSK in left_color_dataset:
            left_dataset[cst.EPI_COLOR_MSK] = xr.DataArray(
                left_color_dataset[cst.EPI_MSK].values, dims=[cst.ROW, cst.COL]
            )

        # if cst.BAND_IM not in left_dataset.dims:
        left_dataset.coords[cst.BAND_IM] = band_im
        left_dataset[cst.EPI_COLOR] = xr.DataArray(
            left_color_dataset[cst.EPI_IMAGE].values,
            dims=[cst.BAND_IM, cst.ROW, cst.COL],
        )

        # Add input color type
        color_type = inputs.rasterio_get_color_type(color1)
        left_dataset[cst.EPI_COLOR].attrs["color_type"] = color_type
    else:
        color_types = inputs.rasterio_get_color_type(img1)
        left_dataset[cst.EPI_IMAGE].attrs["color_type"] = color_types

    # Add classification layers to dataset
    if left_classif_dataset:
        left_dataset.coords[cst.BAND_CLASSIF] = left_classif_dataset.attrs[
            cst.BAND_NAMES
        ]
        left_dataset[cst.EPI_CLASSIFICATION] = xr.DataArray(
            left_classif_dataset[cst.EPI_IMAGE].values,
            dims=[cst.BAND_CLASSIF, cst.ROW, cst.COL],
        )
    if right_classif_dataset:
        right_dataset.coords[cst.BAND_CLASSIF] = right_classif_dataset.attrs[
            cst.BAND_NAMES
        ]
        right_dataset[cst.EPI_CLASSIFICATION] = xr.DataArray(
            right_classif_dataset[cst.EPI_IMAGE].values,
            dims=[cst.BAND_CLASSIF, cst.ROW, cst.COL],
        )
    # Add attributes info
    attributes = {}
    # fill datasets with saving info, window, profile, overlaps for correct
    #  saving
    cars_dataset.fill_dataset(
        left_dataset,
        saving_info=saving_info_left,
        window=window,
        profile=None,
        attributes=attributes,
        overlaps=left_overlaps,
    )

    cars_dataset.fill_dataset(
        right_dataset,
        saving_info=saving_info_right,
        window=window,
        profile=None,
        attributes=attributes,
        overlaps=right_overlaps,
    )

    return left_dataset, right_dataset
