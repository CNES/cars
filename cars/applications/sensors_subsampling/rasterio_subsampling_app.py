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
import copy
import os
from pathlib import Path

# Standard imports
from typing import Dict, Tuple

import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from json_checker import Checker

import cars.orchestrator.orchestrator as ocht
from cars.applications.resampling import (
    resampling_wrappers,
)
from cars.applications.sensors_subsampling import (
    abstract_subsampling_app as ssa,
)
from cars.applications.sensors_subsampling import (
    subsampling_algo,
)
from cars.core import constants as cst
from cars.core import tiling
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset, format_transformation
from cars.pipelines.parameters import sensor_inputs_constants as sens_cst

# pylint: disable= C0302


class RasterioSubsampling(ssa.SensorsSubsampling, short_name=["rasterio"]):
    """
    SensorsSubsampling
    """

    def __init__(self, conf=None):
        """
        Init function of SensorsSubsampling

        :param conf: configuration for SensorsSubsampling
        :return: an application_to_use object
        """

        super().__init__(conf=conf)
        # check conf
        self.used_method = self.used_config["method"]

        self.tile_size = self.used_config["tile_size"]

        # Saving files
        self.save_intermediate_data = self.used_config["save_intermediate_data"]

        self.interpolator_image = self.used_config["interpolator_image"]
        self.interpolator_classif = self.used_config["interpolator_classif"]
        self.interpolator_mask = self.used_config["interpolator_mask"]

        self.overlap = self.used_config["overlap"]

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
        overloaded_conf["method"] = conf.get("method", "rasterio")

        overloaded_conf["tile_size"] = conf.get("tile_size", 10000)

        # Saving files
        overloaded_conf["save_intermediate_data"] = conf.get(
            "save_intermediate_data", False
        )

        overloaded_conf["interpolator_image"] = conf.get(
            "interpolator_image", "bilinear"
        )
        overloaded_conf["interpolator_classif"] = conf.get(
            "interpolator_image", "nearest"
        )
        overloaded_conf["interpolator_mask"] = conf.get(
            "interpolator_image", "nearest"
        )

        overloaded_conf["overlap"] = conf.get("overlap", 2)

        triangulation_schema = {
            "method": str,
            "tile_size": int,
            "save_intermediate_data": bool,
            "interpolator_image": str,
            "interpolator_mask": str,
            "interpolator_classif": str,
            "overlap": int,
        }

        # Check conf
        checker = Checker(triangulation_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def margins_fun(  # pylint: disable=unused-argument
        self, row_min, row_max, col_min, col_max
    ):
        """
        Default margin function, returning zeros
        """
        corner = ["left", "up", "right", "down"]
        data = [self.overlap] * len(corner)
        col = np.arange(len(corner))
        margins = xr.Dataset(
            {"left_margin": (["col"], data)}, coords={"col": col}
        )
        margins["right_margin"] = xr.DataArray(data, dims=["col"])
        return margins

    def update_profile(self, img_path, scale_factor):
        """
        Update rasterio profile

        :param img_path: the image path
        :type img_path: str
        :param scale_factor: the scaling factor
        :type scale_factor: float
        """
        with rasterio.open(img_path) as src:
            height = src.height
            width = src.width

            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)

            transform = src.transform
            new_transform = transform * Affine.scale(
                (width / new_width), (height / new_height)
            )

            profile = src.profile.copy()
            profile.update(
                {
                    "height": new_height,
                    "width": new_width,
                    "transform": new_transform,
                    "driver": "GTiff",
                }
            )

        return profile, height, width

    def get_paths_dictionary(self, sensor_dict):
        """
        Get the paths dictionary

        :param sensor_dict: the sensor dictionary (classification, image...)
        :type sensor_dict: dict
        """
        # get images, no data and classifs
        image_dict = resampling_wrappers.get_paths_and_bands_from_image(
            sensor_dict[sens_cst.INPUT_IMG],
            None,
        )

        image = next(iter(image_dict))

        mask = sensor_dict.get(sens_cst.INPUT_MSK, None)

        classif = sensor_dict.get(sens_cst.INPUT_CLASSIFICATION, None)
        classif_path = None
        if classif is not None:
            classif = resampling_wrappers.get_path_and_values_from_classif(
                classif
            )

            classif_path = next(iter(classif))

        paths_dictionary = {"im": image, "mask": mask, "classif": classif_path}

        step = 0
        for key in list(image_dict.keys())[1:]:
            paths_dictionary["texture_" + str(step)] = key
            step += 1

        return paths_dictionary

    def run(  # pylint: disable=too-many-positional-arguments
        self,
        id_image,
        sensor_dict,
        resolution,
        out_directory,
        orchestrator,
    ):
        """
        Run subsampling using rasterio

        :param id_image: the id of the image
        :type id_image: str
        :param sensor_dict: the sensor dictionnary (image, classification...)
        :type sensor_dict: dict
        :param resolution: the subsampling resolution
        :type resolution: int
        :param out_directory: the output directory
        :type out_directory: str
        :param orchestrator: orchestrator used
        """

        # Default orchestrator
        if orchestrator is None:
            # Create default sequential orchestrator for current application
            # be awere, no out_json will be shared between orchestrators
            # No files saved
            self.orchestrator = ocht.Orchestrator(
                orchestrator_conf={"mode": "sequential"}
            )
        else:
            self.orchestrator = orchestrator

        # Create CarsDataset
        # images
        image_subsampled = cars_dataset.CarsDataset(
            "arrays", name="subsampling_left_" + str(resolution)
        )

        # Define the scale factor
        scale_factor = 1 / resolution

        # Define the path of each image
        img_path = sensor_dict[sens_cst.INPUT_IMG]["bands"]["b0"]["path"]

        # update profile of each image
        new_profile, height, width = self.update_profile(img_path, scale_factor)

        # Get saving infos in order to save tiles when they are computed
        [
            saving_info,
        ] = self.orchestrator.get_saving_infos([image_subsampled])

        # Build paths dictionnary
        paths_dictionary = self.get_paths_dictionary(sensor_dict)

        # Define tiling grid
        tiling_grid = tiling.generate_tiling_grid(
            0,
            0,
            height,
            width,
            self.tile_size,
            self.tile_size,
        )

        # Compute tiling grid
        image_subsampled.tiling_grid = tiling_grid

        # Save files
        safe_makedirs(os.path.join(out_directory, id_image))
        for key, path in paths_dictionary.items():
            if key == "classif":
                dtype = np.uint8
                optional_data = True
                nodata = 255
            elif key == "mask":
                dtype = np.uint8
                nodata = 0
                optional_data = False
            else:
                dtype = "float32"
                nodata = 0
                optional_data = False

            if path is not None:
                name_file = Path(path).name
                orchestrator.add_to_save_lists(
                    os.path.join(out_directory, id_image, name_file),
                    key,
                    image_subsampled,
                    dtype=dtype,
                    nodata=nodata,
                    optional_data=optional_data,
                )

        # Compute overlaps
        (
            image_subsampled.overlaps,
            _,
            _,
            _,
        ) = format_transformation.grid_margins_2_overlaps(
            image_subsampled.tiling_grid, self.margins_fun
        )

        # Generate Image pair
        for col in range(image_subsampled.shape[1]):
            for row in range(image_subsampled.shape[0]):

                # update saving infos  for potential replacement
                full_saving_info = ocht.update_saving_infos(
                    saving_info, row=row, col=col
                )
                overlap = cars_dataset.overlap_array_to_dict(
                    image_subsampled.overlaps[row, col]
                )

                window = image_subsampled.get_window_as_dict(row, col)
                # Compute images
                (
                    image_subsampled[row, col]
                ) = self.orchestrator.cluster.create_task(
                    generate_subsampled_images_wrapper, nout=1
                )(
                    paths_dictionary,
                    new_profile,
                    scale_factor,
                    self.tile_size,
                    window=window,
                    saving_info=full_saving_info,
                    overlap=overlap,
                )
        return image_subsampled


# pylint: disable=too-many-positional-arguments
def generate_subsampled_images_wrapper(
    paths_dictionary,
    profile,
    scale_factor=1,
    tile_size=10000,
    window=None,
    saving_info=None,
    overlap=None,
) -> Dict[str, Tuple[xr.Dataset, xr.Dataset]]:
    """
    Subsampling wrapper

    :param paths_dictionary: the paths dictionary
    :type paths_dictionary: dict
    :param profile: the new profile
    :param scale_factor: the scaling_factor
    :type scale_factor: float
    :param tile_size: the tile size
    :type tile_size: int
    :param window: the current window
    :type window: Window
    :param saving_info: the saving information
    :param overlap: the overlap
    """

    global_dataset = None
    for key, path in paths_dictionary.items():
        # Rectify images
        if path is not None:
            interpolator = "bilinear"
            if key in ("classif", "mask"):
                interpolator = "nearest"

            dataset = subsampling_algo.resample_image(
                path,
                window,
                tile_size,
                key,
                scale_factor=scale_factor,
                interpolator=interpolator,
            )
            if key == "im":
                global_dataset = copy.deepcopy(dataset)

            if key == "classif":
                global_dataset[key] = xr.DataArray(
                    dataset[cst.EPI_IMAGE].values,
                    dims=[cst.ROW, cst.COL],
                )

            if key == "mask":
                global_dataset[key] = xr.DataArray(
                    dataset[cst.EPI_IMAGE].values,
                    dims=[cst.ROW, cst.COL],
                )

            if "texture" in key:
                global_dataset.coords[cst.BAND_IM] = dataset.attrs[
                    cst.BAND_NAMES
                ]
                global_dataset[key] = xr.DataArray(
                    dataset[cst.EPI_IMAGE].values,
                    dims=[cst.BAND_NAMES, cst.ROW, cst.COL],
                )

    window_out_left = {
        "row_min": global_dataset.region[1],
        "row_max": global_dataset.region[3],
        "col_min": global_dataset.region[0],
        "col_max": global_dataset.region[2],
    }

    # Add attributes info
    attributes = {}
    # fill datasets with saving info, window, profile, overlaps for correct
    #  saving
    cars_dataset.fill_dataset(
        global_dataset,
        saving_info=saving_info,
        window=window_out_left,
        profile=profile,
        attributes=attributes,
        overlaps=overlap,
    )

    return global_dataset
