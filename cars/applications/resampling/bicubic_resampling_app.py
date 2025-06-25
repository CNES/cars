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
this module contains the bicubic_resampling application class.
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
from json_checker import And, Checker, Or
from shapely.geometry import Polygon

# CARS imports
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.resampling import (
    resampling_algo,
    resampling_constants,
    resampling_wrappers,
)
from cars.applications.resampling.abstract_resampling_app import Resampling
from cars.core import constants as cst
from cars.core import inputs, tiling
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset, format_transformation
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst


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
        self.strip_height = self.used_config["strip_height"]
        self.step = self.used_config["step"]

        # Saving bools
        self.save_intermediate_data = self.used_config["save_intermediate_data"]

        self.interpolator_image = self.used_config["interpolator_image"]
        self.interpolator_classif = self.used_config["interpolator_classif"]
        self.interpolator_mask = self.used_config["interpolator_mask"]

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
        overloaded_conf["strip_height"] = conf.get("strip_height", 60)
        overloaded_conf["interpolator_image"] = conf.get(
            "interpolator_image", "bicubic"
        )
        overloaded_conf["interpolator_classif"] = conf.get(
            "interpolator_classif", "nearest"
        )
        overloaded_conf["interpolator_mask"] = conf.get(
            "interpolator_mask", "nearest"
        )
        overloaded_conf["step"] = conf.get("step", 500)

        # Saving bools
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        rectification_schema = {
            "method": str,
            "strip_height": And(int, lambda x: x > 0),
            "interpolator_image": str,
            "interpolator_classif": str,
            "interpolator_mask": str,
            "step": Or(None, int),
            "save_intermediate_data": bool,
        }

        # Check conf
        checker = Checker(rectification_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def pre_run(
        self,
        grid_left,
        tile_width,
        tile_height,
    ):
        """
        Pre run some computations : tiling grid

        :param grid_left: left grid
        :type grid_left: dict
        :param optimum_tile_size: optimum tile size
        :type optimum_tile_size: int


        :return: epipolar_regions_grid, epipolar_regions,
            opt_epipolar_tile_size, largest_epipolar_region,
        """

        # Get largest epipolar regions from configuration file
        largest_epipolar_region = [
            0,
            0,
            grid_left["epipolar_size_x"],
            grid_left["epipolar_size_y"],
        ]

        origin = grid_left["grid_origin"]
        spacing = grid_left["grid_spacing"]

        logging.info(
            "Size of epipolar image: {}".format(largest_epipolar_region)
        )
        logging.debug("Origin of epipolar grid: {}".format(origin))
        logging.debug("Spacing of epipolar grid: {}".format(spacing))

        if tile_width is None:
            tile_width = grid_left["epipolar_size_x"]
        if tile_height is None:
            tile_height = self.strip_height

        logging.info(
            "Tile size for epipolar regions: "
            "{width}x{height} pixels".format(
                width=tile_width, height=tile_height
            )
        )

        epipolar_regions_grid = tiling.generate_tiling_grid(
            0,
            0,
            grid_left["epipolar_size_y"],
            grid_left["epipolar_size_x"],
            tile_height,
            tile_width,
        )

        logging.info(
            "Epipolar image will be processed in {} splits".format(
                epipolar_regions_grid.shape[0] * epipolar_regions_grid.shape[1]
            )
        )

        return (
            epipolar_regions_grid,
            tile_width,
            tile_height,
            largest_epipolar_region,
        )

    def run(  # noqa: C901
        self,
        sensor_image_left,
        sensor_image_right,
        grid_left,
        grid_right,
        geom_plugin,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
        margins_fun=None,
        tile_width=None,
        tile_height=None,
        add_classif=True,
        epipolar_roi=None,
        required_bands=None,
        texture_bands=None,
    ):
        """
        Run resampling application.

        Creates left and right CarsDataset filled with xarray.Dataset,
        corresponding to sensor images resampled in epipolar geometry.

        :param sensor_images_left: tiled sensor left image
            Dict Must contain keys : "image", "geomodel",
            "no_data", "mask", "classification". Paths must be absolutes
        :type sensor_images_left: CarsDataset
        :param sensor_images_right: tiled sensor right image
            Dict Must contain keys : "image", "geomodel",
            "no_data", "mask", "classification". Paths must be absolutes
        :type sensor_images_right: CarsDataset
        :param grid_left: left epipolar grid
            Grid dict contains :
            - "grid_spacing", "grid_origin", \
                "epipolar_size_x", "epipolar_size_y", "epipolar_origin_x",\
                 "epipolar_origin_y", epipolar_spacing_x",\
                 "epipolar_spacing", "disp_to_alt_ratio", "path"
        :type grid_left: dict
        :param grid_right: right epipolar grid. Grid dict contains :
            - "grid_spacing", "grid_origin",\
                "epipolar_size_x", "epipolar_size_y", "epipolar_origin_x",\
                 "epipolar_origin_y", epipolar_spacing_x",\
                 "epipolar_spacing", "disp_to_alt_ratio", "path"
        :type grid_right: dict
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: directory to save files to
        :param pair_key: pair id
        :type pair_key: str
        :param margins_fun: margins function to use
        :type margins_fun: fun
        :param optimum_tile_size: optimum tile size to use
        :type optimum_tile_size: int
        :param tile_width: width of tile
        :type tile_width: int
        :param tile_height: height of tile
        :type tile_height: int
        :param add_classif: add classif to dataset
        :type add_classif: bool
        :param epipolar_roi: Epipolar roi to use if set.
            Set None tiles outsize roi
        :type epipolar_roi: list(int), [row_min, row_max,  col_min, col_max]
        :param required_bands: bands to resample on left and right image
        :type required_bands: dict
        :param texture_bands: name of bands used for output texture
        :type texture_bands: list

        :return: left epipolar image, right epipolar image. \
            Each CarsDataset contains:

            - N x M Delayed tiles. \
                Each tile will be a future xarray Dataset containing:

                - data with keys : "im", "msk", "classif"
                - attrs with keys: "margins" with "disp_min" and "disp_max"\
                    "transform", "crs", "valid_pixels", "no_data_mask",
                    "no_data_img"
            - attributes containing: \
                "largest_epipolar_region","opt_epipolar_tile_size",
                "disp_min_tiling", "disp_max_tiling"

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

        # Create zeros margins if not provided
        if margins_fun is None:

            def margins_fun(  # pylint: disable=unused-argument
                row_min, row_max, col_min, col_max
            ):
                """
                Default margin function, returning zeros
                """
                corner = ["left", "up", "right", "down"]
                data = np.zeros(len(corner))
                col = np.arange(len(corner))
                margins = xr.Dataset(
                    {"left_margin": (["col"], data)}, coords={"col": col}
                )
                margins["right_margin"] = xr.DataArray(data, dims=["col"])
                return margins

        # Get grids and regions for current pair
        (
            epipolar_regions_grid,
            tile_width,
            tile_height,
            largest_epipolar_region,
        ) = self.pre_run(
            grid_left,
            tile_width,
            tile_height,
        )

        # Create CarsDataset
        # Epipolar_images
        epipolar_images_left = cars_dataset.CarsDataset(
            "arrays", name="resampling_left_" + pair_key
        )
        epipolar_images_right = cars_dataset.CarsDataset(
            "arrays", name="resampling_" + pair_key
        )

        # Compute tiling grid
        epipolar_images_left.tiling_grid = epipolar_regions_grid

        # Generate tiling grid
        epipolar_images_right.tiling_grid = epipolar_regions_grid

        # Compute overlaps
        (
            epipolar_images_left.overlaps,
            epipolar_images_right.overlaps,
            used_disp_min,
            used_disp_max,
        ) = format_transformation.grid_margins_2_overlaps(
            epipolar_images_left.tiling_grid, margins_fun
        )

        # add image type in attributes for future checking
        if texture_bands is not None:
            im_type = inputs.rasterio_get_image_type(
                sensor_image_left[sens_cst.INPUT_IMG]["bands"][
                    texture_bands[0]
                ]["path"]
            )
        else:
            im_type = inputs.rasterio_get_image_type(
                sensor_image_left[sens_cst.INPUT_IMG]["main_file"]
            )

        # update attributes
        epipolar_images_attributes = {
            "largest_epipolar_region": largest_epipolar_region,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "disp_min_tiling": used_disp_min,
            "disp_max_tiling": used_disp_max,
            "image_type": im_type,
        }

        epipolar_images_left.attributes.update(epipolar_images_attributes)
        epipolar_images_right.attributes.update(epipolar_images_attributes)

        # Save objects
        if self.save_intermediate_data:
            safe_makedirs(pair_folder)

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
                dtype=np.uint8,
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epi_img_right_mask.tif"),
                cst.EPI_MSK,
                epipolar_images_right,
                cars_ds_name="epi_img_right_mask",
                dtype=np.uint8,
            )

        if self.save_intermediate_data and add_classif:
            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epi_img_left_classif.tif"),
                cst.EPI_CLASSIFICATION,
                epipolar_images_left,
                cars_ds_name="epi_img_left_classif",
                dtype=np.uint8,
                optional_data=True,
            )

            self.orchestrator.add_to_save_lists(
                os.path.join(pair_folder, "epi_img_right_classif.tif"),
                cst.EPI_CLASSIFICATION,
                epipolar_images_right,
                cars_ds_name="epi_img_right_classif",
                dtype=np.uint8,
                optional_data=True,
            )

        # Get saving infos in order to save tiles when they are computed
        [
            saving_info_left,
            saving_info_right,
        ] = self.orchestrator.get_saving_infos(
            [epipolar_images_left, epipolar_images_right]
        )

        logging.info(
            "Number of tiles in epipolar resampling: "
            "row: {} "
            "col: {}".format(
                epipolar_images_left.tiling_grid.shape[0],
                epipolar_images_left.tiling_grid.shape[1],
            )
        )

        # Add infos to orchestrator.out_json
        updating_dict = {
            application_constants.APPLICATION_TAG: {
                resampling_constants.RESAMPLING_RUN_TAG: {
                    pair_key: {resampling_constants.METHOD: self.used_method},
                }
            }
        }
        self.orchestrator.update_out_info(updating_dict)

        # retrieve data
        epipolar_size_x = grid_left.attributes["epipolar_size_x"]
        epipolar_size_y = grid_left.attributes["epipolar_size_y"]
        left_images = resampling_wrappers.get_paths_and_bands(
            sensor_image_left[sens_cst.INPUT_IMG],
            required_bands["left"],
        )
        right_images = resampling_wrappers.get_paths_and_bands(
            sensor_image_right[sens_cst.INPUT_IMG],
            required_bands["right"],
        )
        grid1 = grid_left
        grid2 = grid_right
        nodata1 = sensor_image_left[sens_cst.INPUT_IMG].get(
            sens_cst.INPUT_NODATA, None
        )
        nodata2 = sensor_image_right[sens_cst.INPUT_IMG].get(
            sens_cst.INPUT_NODATA, None
        )
        mask1 = sensor_image_left.get(sens_cst.INPUT_MSK, None)
        mask2 = sensor_image_right.get(sens_cst.INPUT_MSK, None)
        left_classifs = sensor_image_left.get(
            sens_cst.INPUT_CLASSIFICATION, None
        )
        if left_classifs is not None:
            left_classifs = resampling_wrappers.get_paths_and_bands(
                left_classifs
            )
        right_classifs = sensor_image_right.get(
            sens_cst.INPUT_CLASSIFICATION, None
        )
        if right_classifs is not None:
            right_classifs = resampling_wrappers.get_paths_and_bands(
                right_classifs
            )

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

        # Check if tiles are in sensors
        in_sensor_left_array, in_sensor_right_array = (
            resampling_wrappers.check_tiles_in_sensor(
                sensor_image_left,
                sensor_image_right,
                epi_tilling_grid,
                grid_left,
                grid_right,
                geom_plugin,
            )
        )

        # broadcast grids
        broadcasted_grid1 = self.orchestrator.cluster.scatter(grid1)
        broadcasted_grid2 = self.orchestrator.cluster.scatter(grid2)

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

                if epipolar_roi_poly.intersects(tile_roi_poly) and (
                    in_sensor_left_array[row, col]
                    or in_sensor_right_array[row, col]
                ):
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
                        epipolar_size_x,
                        epipolar_size_y,
                        left_images,
                        right_images,
                        broadcasted_grid1,
                        broadcasted_grid2,
                        self.interpolator_image,
                        self.interpolator_classif,
                        self.interpolator_mask,
                        self.step,
                        used_disp_min=used_disp_min[row, col],
                        used_disp_max=used_disp_max[row, col],
                        add_classif=add_classif,
                        mask1=mask1,
                        mask2=mask2,
                        left_classifs=left_classifs,
                        right_classifs=right_classifs,
                        nodata1=nodata1,
                        nodata2=nodata2,
                        saving_info_left=full_saving_info_left,
                        saving_info_right=full_saving_info_right,
                    )

                    # Remove tile with all nan
                    if not in_sensor_left_array[row, col]:
                        epipolar_images_left[row, col] = None
                    if not in_sensor_right_array[row, col]:
                        epipolar_images_right[row, col] = None
        return epipolar_images_left, epipolar_images_right


def generate_epipolar_images_wrapper(
    left_overlaps,
    right_overlaps,
    window,
    epipolar_size_x,
    epipolar_size_y,
    left_imgs,
    right_imgs,
    grid1,
    grid2,
    interpolator_image,
    interpolator_classif,
    interpolator_mask,
    step=None,
    used_disp_min=None,
    used_disp_max=None,
    add_classif=True,
    mask1=None,
    mask2=None,
    left_classifs=None,
    right_classifs=None,
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

    :return: Left image object, Right image object (if exists)

    Returned objects are composed of dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_TEXTURE (for left, if given)
    """

    region, margins = format_transformation.region_margins_from_window(
        window,
        left_overlaps,
        right_overlaps,
        used_disp_min=used_disp_min,
        used_disp_max=used_disp_max,
    )

    # Rectify images
    (
        left_dataset,
        right_dataset,
        left_classif_dataset,
        right_classif_dataset,
    ) = resampling_algo.epipolar_rectify_images(
        left_imgs,
        right_imgs,
        grid1,
        grid2,
        region,
        margins,
        epipolar_size_x,
        epipolar_size_y,
        interpolator_image,
        interpolator_classif,
        interpolator_mask,
        step=step,
        mask1=mask1,
        mask2=mask2,
        left_classifs=left_classifs,
        right_classifs=right_classifs,
        nodata1=nodata1,
        nodata2=nodata2,
        add_classif=add_classif,
    )

    # Add classification layers to dataset
    if add_classif:
        if left_classif_dataset:
            left_dataset.coords[cst.BAND_CLASSIF] = left_classif_dataset.attrs[
                cst.BAND_NAMES
            ]
            left_dataset[cst.EPI_CLASSIFICATION] = xr.DataArray(
                left_classif_dataset[cst.EPI_IMAGE].values,
                dims=[cst.BAND_CLASSIF, cst.ROW, cst.COL],
            ).astype(bool)
        if right_classif_dataset:
            right_dataset.coords[cst.BAND_CLASSIF] = (
                right_classif_dataset.attrs[cst.BAND_NAMES]
            )
            right_dataset[cst.EPI_CLASSIFICATION] = xr.DataArray(
                right_classif_dataset[cst.EPI_IMAGE].values,
                dims=[cst.BAND_CLASSIF, cst.ROW, cst.COL],
            ).astype(bool)

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
