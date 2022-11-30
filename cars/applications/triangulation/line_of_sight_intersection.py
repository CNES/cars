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

# Standard imports
import logging
import os
from typing import Dict, Tuple

import pandas

# Third party imports
import xarray as xr
from json_checker import Checker

import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.grid_generation import grids
from cars.applications.triangulation import (
    triangulation_constants,
    triangulation_tools,
)
from cars.applications.triangulation.triangulation import Triangulation
from cars.core import constants as cst
from cars.core import preprocessing

# CARS imports
from cars.core.geometry import AbstractGeometry, read_geoid_file
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.pipelines.sensor_to_full_resolution_dsm import (
    sensor_full_res_dsm_constants as sens_cst,
)


class LineOfSightIntersection(
    Triangulation, short_name="line_of_sight_intersection"
):
    """
    Triangulation
    """

    def __init__(self, conf=None):
        """
        Init function of Triangulation

        :param conf: configuration for triangulation
        :return: an application_to_use object
        """

        super().__init__(conf=conf)
        # check conf
        self.used_method = self.used_config["method"]
        self.use_geoid_alt = self.used_config["use_geoid_alt"]
        self.snap_to_img1 = self.used_config["snap_to_img1"]
        self.add_msk_info = self.used_config["add_msk_info"]
        # Saving files
        self.save_points_cloud = self.used_config["save_points_cloud"]

        # check loader
        # TODO
        self.geometry_loader = self.used_config["geometry_loader"]
        AbstractGeometry(  # pylint: disable=abstract-class-instantiated
            self.geometry_loader
        )

        # global value for left image to check if snap_to_img1 can
        # be applied : Need than same application object is run
        # for all pairs
        self.ref_left_img = None

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
        overloaded_conf["method"] = conf.get(
            "method", "line_of_sight_intersection"
        )
        overloaded_conf["use_geoid_alt"] = conf.get("use_geoid_alt", False)
        overloaded_conf["snap_to_img1"] = conf.get("snap_to_img1", False)
        overloaded_conf["add_msk_info"] = conf.get("add_msk_info", True)
        # Overloader loader
        overloaded_conf["geometry_loader"] = conf.get(
            "geometry_loader", "OTBGeometry"
        )
        # Saving files
        overloaded_conf["save_points_cloud"] = conf.get(
            "save_points_cloud", False
        )

        tringulation_schema = {
            "method": str,
            "use_geoid_alt": bool,
            "snap_to_img1": bool,
            "add_msk_info": bool,
            "save_points_cloud": bool,
            "geometry_loader": str,
        }

        # Check conf
        checker = Checker(tringulation_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def get_geometry_loader(self):

        return self.geometry_loader

    def run(
        self,
        sensor_image_left,
        sensor_image_right,
        epipolar_images_left,
        epipolar_images_right,
        grid_left,
        grid_right,
        epipolar_disparity_map_left,
        epipolar_disparity_map_right,
        epsg,
        orchestrator=None,
        pair_folder=None,
        pair_key="PAIR_0",
        uncorrected_grid_right=None,
        geoid_path=None,
        disp_min=0,  # used for corresponding tiles in fusion pre processing
        disp_max=0,  # TODO remove
    ):
        """
        Run Triangulation application.

        Created left and right CarsDataset filled with xarray.Dataset,
        corresponding to 3D points clouds, stored on epipolar geometry grid.

        :param sensor_image_left: tiled sensor left image
            Dict Must contain keys : "image", "color", "geomodel",
            "no_data", "mask". Paths must be absolutes
        :type sensor_image_left: CarsDataset
        :param sensor_image_right: tiled sensor right image
            Dict Must contain keys : "image", "color", "geomodel",
            "no_data", "mask". Paths must be absolutes
        :type sensor_image_right: CarsDataset
        :param epipolar_images_left: tiled epipolar left image
        :type epipolar_images_left: CarsDataset
        :param epipolar_images_right: tiled epipolar right image
        :type epipolar_images_right: CarsDataset
        :param grid_left: left epipolar grid. Grid CarsDataset contains :

            - A single tile stored in [0,0], containing a (N, M, 2) shape \
                array in xarray Dataset
            - Attributes containing: "grid_spacing", "grid_origin",\
                "epipolar_size_x", epipolar_size_y", "epipolar_origin_x",\
                "epipolar_origin_y","epipolar_spacing_x",\
                "epipolar_spacing", "disp_to_alt_ratio",\
        :type grid_left: CarsDataset
        :param grid_right: right epipolar grid. Grid CarsDataset contains :

            - A single tile stored in [0,0], containing a (N, M, 2) shape
                array in xarray Dataset
            - Attributes containing: "grid_spacing", "grid_origin",
                "epipolar_size_x", epipolar_size_y", "epipolar_origin_x",
                "epipolar_origin_y","epipolar_spacing_x",
                "epipolar_spacing", "disp_to_alt_ratio",
        :type grid_right: CarsDataset
        :param epipolar_disparity_map_left: tiled left disparity map or \
            sparse matches:

            - if CarsDataset is instance of "arrays", CarsDataset contains:

                - N x M Delayed tiles \
                    Each tile will be a future xarray Dataset containing:

                    - data with keys : "disp", "disp_msk"
                    - attrs with keys: profile, window, overlaps
                - attributes containing:"largest_epipolar_region"\
                  "opt_epipolar_tile_size","epipolar_regions_grid"

            - if CarsDataset is instance of "points", CarsDataset contains:

                - N x M Delayed tiles \
                    Each tile will be a future pandas DataFrame containing:

                    - data : (L, 4) shape matches
                - attributes containing:"disp_lower_bound","disp_upper_bound",\
                    "elevation_delta_lower_bound","elevation_delta_upper_bound"

        :type epipolar_disparity_map_left: CarsDataset
        :param epipolar_disparity_map_right: tiled right disparity map or
             sparse matches
        :type epipolar_disparity_map_right: CarsDataset
        :param orchestrator: orchestrator used
        :param pair_folder: folder used for current pair
        :type pair_folder: str
        :param pair_key: pair key id
        :type pair_key: str
        :param uncorrected_grid_right: not corrected right epipolar grid
                used if self.snap_to_img1
        :type uncorrected_grid_right: CarsDataset
        :param geoid_path: geoid path
        :type geoid_path: str
        :param disp_min: minimum disparity
        :type disp_min: int
        :param disp_max: maximum disparity
        :type disp_max: int

        :return: left points cloud, right points cloud. \
                 Each CarsDataset contains:

            - N x M Delayed tiles \
                Each tile will be a future xarray Dataset containing:

                - data : with keys : "x", "y", "z", "corr_msk"\
                    optional: "color", "msk",
                - attrs with keys: "margins", "epi_full_size", "epsg"
            - attributes containing: "disp_lower_bound",  "disp_upper_bound", \
                "elevation_delta_lower_bound","elevation_delta_upper_bound"

        :rtype: Tuple(CarsDataset, CarsDataset)
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

        if pair_folder is None:
            pair_folder = os.path.join(self.orchestrator.out_dir, "tmp")
            safe_makedirs(pair_folder)

        # Get local conf left image for this in_json iteration
        conf_left_img = sensor_image_left[sens_cst.INPUT_IMG]
        # Check left image and raise a warning
        # if different left images are used along with snap_to_img1 mode
        if self.ref_left_img is None:
            self.ref_left_img = conf_left_img
        else:
            if self.snap_to_img1 and self.ref_left_img != conf_left_img:
                logging.warning(
                    "snap_to_left_image mode is used but inputs "
                    " have different images as their "
                    "left image in pair. This may result in "
                    "increasing registration discrepancies between pairs"
                )

        # Add log about geoid
        alt_reference = None
        if self.use_geoid_alt:
            if geoid_path is not None:
                alt_reference = "geoid"
        else:
            alt_reference = "ellipsoid"

        # Add infos to orchestrator.out_json
        updating_dict = {
            application_constants.APPLICATION_TAG: {
                pair_key: {
                    triangulation_constants.TRIANGULATION_RUN_TAG: {
                        triangulation_constants.ALT_REFERENCE_TAG: (
                            alt_reference
                        )
                    },
                }
            }
        }
        self.orchestrator.update_out_info(updating_dict)

        # Create fake configuration dict
        # TODO remove it later
        configuration = (
            preprocessing.create_former_cars_post_prepare_configuration(
                sensor_image_left,
                sensor_image_right,
                grid_left,
                grid_right,
                pair_folder,
                uncorrected_grid_right=uncorrected_grid_right,
                srtm_dir=None,
                default_alt=0,
                disp_min=None,
                disp_max=None,
            )
        )

        # Compute disp_min and disp_max location for epipolar grid
        (
            epipolar_grid_min,
            epipolar_grid_max,
        ) = grids.compute_epipolar_grid_min_max(
            self.geometry_loader,
            epipolar_images_left.attributes["epipolar_regions_grid"],
            epsg,
            configuration,
            disp_min,
            disp_max,
        )
        # update attributes for corresponding tiles in cloud fusion
        # TODO remove with refactoring
        pc_attributes = {
            "used_epsg_for_terrain_grid": epsg,
            "epipolar_regions_grid": epipolar_images_left.attributes[
                "epipolar_regions_grid"
            ],
            "epipolar_grid_min": epipolar_grid_min,
            "epipolar_grid_max": epipolar_grid_max,
            "largest_epipolar_region": epipolar_images_left.attributes[
                "largest_epipolar_region"
            ],
            "opt_epipolar_tile_size": epipolar_images_left.attributes[
                "opt_epipolar_tile_size"
            ],
        }

        epipolar_points_cloud_left, epipolar_points_cloud_right = None, None

        if epipolar_disparity_map_left.dataset_type in ("arrays", "points"):
            # Create CarsDataset
            # Epipolar_point_cloud
            epipolar_points_cloud_left = cars_dataset.CarsDataset("arrays")
            epipolar_points_cloud_left.create_empty_copy(epipolar_images_left)
            epipolar_points_cloud_left.overlaps *= 0  # Margins removed

            epipolar_points_cloud_right = cars_dataset.CarsDataset("arrays")
            epipolar_points_cloud_right.create_empty_copy(epipolar_images_right)

            # Update attributes to get epipolar info
            epipolar_points_cloud_left.attributes.update(pc_attributes)

            # Save objects
            if self.save_points_cloud:
                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_pc_X_left.tif"),
                    cst.X,
                    epipolar_points_cloud_left,
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_pc_Y_left.tif"),
                    cst.Y,
                    epipolar_points_cloud_left,
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_pc_Z_left.tif"),
                    cst.Z,
                    epipolar_points_cloud_left,
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_pc_X_right.tif"),
                    cst.X,
                    epipolar_points_cloud_right,
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_pc_Y_right.tif"),
                    cst.Y,
                    epipolar_points_cloud_right,
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_pc_Z_right.tif"),
                    cst.Z,
                    epipolar_points_cloud_right,
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_pc_color_left.tif"),
                    cst.EPI_COLOR,
                    epipolar_points_cloud_left,
                )

                self.orchestrator.add_to_save_lists(
                    os.path.join(pair_folder, "epi_pc_color_right.tif"),
                    cst.EPI_COLOR,
                    epipolar_points_cloud_right,
                )

        else:
            logging.error(
                "Triangulation application doesn't "
                "support this input data format"
            )

        # Get saving infos in order to save tiles when they are computed
        [
            saving_info_left,
            saving_info_right,
        ] = self.orchestrator.get_saving_infos(
            [epipolar_points_cloud_left, epipolar_points_cloud_right]
        )

        # Generate Point clouds
        # Broadcast geoid_data
        geoid_data_futures = None
        if self.use_geoid_alt:
            if geoid_path is None:
                logging.error(
                    "use_geoid_alt option is activated but no geoid file "
                    "has been defined in inputs"
                )

            geoid_data = read_geoid_file(geoid_path)
            # Broadcast geoid data to all dask workers
            geoid_data_futures = self.orchestrator.cluster.scatter(
                geoid_data, broadcast=True
            )

        for col in range(epipolar_images_left.shape[1]):
            for row in range(epipolar_images_left.shape[0]):

                # Compute points
                (
                    epipolar_points_cloud_left[row][col],
                    epipolar_points_cloud_right[row][col],
                ) = self.orchestrator.cluster.create_task(
                    compute_points_cloud, nout=2
                )(
                    epipolar_images_left[row, col],
                    epipolar_images_right[row, col],
                    epipolar_disparity_map_left[row, col],
                    epipolar_disparity_map_right[row, col],
                    configuration,
                    self.geometry_loader,
                    geoid_data=geoid_data_futures,
                    snap_to_img1=self.snap_to_img1,
                    add_msk_info=self.add_msk_info,
                    saving_info_left=saving_info_left,
                    saving_info_right=saving_info_right,
                )

        return epipolar_points_cloud_left, epipolar_points_cloud_right


def compute_points_cloud(
    left_image_object: xr.Dataset,
    right_image_object: xr.Dataset,
    left_disparity_object: xr.Dataset,
    right_disparity_object: xr.Dataset,
    input_stereo_cfg: dict,
    geometry_loader: str,
    geoid_data: xr.Dataset = None,
    snap_to_img1: bool = False,
    add_msk_info: bool = False,
    saving_info_left=None,
    saving_info_right=None,
) -> Dict[str, Tuple[xr.Dataset, xr.Dataset]]:
    """
    Compute points clouds from image objects and disparity objects.

    :param left_image_object: Left image dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_COLOR (for left, if given)
    :type left_image_object: xr.Dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_COLOR (for left, if given)
    :param right_image_object: Right image
    :type right_image_object: xr.Dataset
    :param left_disparity_object: Left disparity map dataset with :

            - cst_disp.MAP
            - cst_disp.VALID
            - cst.EPI_COLOR
    :type left_disparity_object: xr.Dataset
    :param right_disparity_object: Right disparity map dataset \
           (None if use_sec_disp not activated) with :

            - cst_disp.MAP
            - cst_disp.VALID
            - cst.EPI_COLOR
    :type right_disparity_object: xr.Dataset
    :param input_stereo_cfg: Configuration for stereo processing
    :type input_stereo_cfg: dict
    :param geometry_loader: name of geometry loader to use
    :type geometry_loader: str
    :param geoid_data: Geoid used for altimetric reference. Defaults to None
        for using ellipsoid as altimetric reference.
    :type geoid_data: str
    :param snap_to_img1: If True, Lines of Sight of img2 are moved so as to
                         cross those of img1
    :type snap_to_img1: bool
    :param add_msk_info:  boolean enabling the addition of the masks'
                         information in the point clouds final dataset
    :type add_msk_info: bool

    :return: Left disparity object, Right disparity object (if exists)

    Returned objects are composed of :

        - dataset (None for right object if use_sec_disp not activated) with :

            - cst.X
            - cst.Y
            - cst.Z
            - cst.EPI_COLOR
    """

    # Get disparity maps
    disp_ref = left_disparity_object
    disp_sec = right_disparity_object

    # Get masks
    left = left_image_object
    right = right_image_object
    im_ref_msk = None
    im_sec_msk = None
    if add_msk_info:
        ref_values_list = [key for key, _ in left.items()]
        if cst.EPI_MSK in ref_values_list:
            im_ref_msk = left
        else:
            worker_logger = logging.getLogger("distributed.worker")
            worker_logger.warning(
                "Left image does not have a mask to rasterize"
            )
        if disp_sec is not None:
            sec_values_list = [key for key, _ in right.items()]
            if cst.EPI_MSK in sec_values_list:
                im_sec_msk = right
            else:
                worker_logger = logging.getLogger("distributed.worker")
                worker_logger.warning(
                    "Right image does not have a mask to rasterize"
                )

    # Triangulate
    if isinstance(disp_ref, xr.Dataset):
        # Triangulate epipolar dense disparities
        if disp_sec is not None:
            points = triangulation_tools.triangulate(
                geometry_loader,
                input_stereo_cfg,
                disp_ref,
                disp_sec,
                snap_to_img1=snap_to_img1,
                im_ref_msk_ds=im_ref_msk,
                im_sec_msk_ds=im_sec_msk,
            )
        else:
            points = triangulation_tools.triangulate(
                geometry_loader,
                input_stereo_cfg,
                disp_ref,
                snap_to_img1=snap_to_img1,
                im_ref_msk_ds=im_ref_msk,
                im_sec_msk_ds=im_sec_msk,
            )
    elif isinstance(disp_ref, pandas.DataFrame):
        # Triangulate epipolar sparse matches
        points = {}
        points[cst.STEREO_REF] = triangulation_tools.triangulate_matches(
            geometry_loader, input_stereo_cfg, disp_ref.to_numpy()
        )

    else:
        logging.error(
            "Disp ref is neither xarray Dataset  nor pandas DataFrame"
        )
        raise Exception(
            "Disp ref is neither xarray Dataset  nor pandas DataFrame"
        )

    if geoid_data is not None:  # if user pass a geoid, use it a alt reference
        for key, point in points.items():
            points[key] = triangulation_tools.geoid_offset(point, geoid_data)

    # Fill datasets
    left_pc_dataset = points[cst.STEREO_REF]
    cars_dataset.fill_dataset(
        left_pc_dataset,
        saving_info=saving_info_left,
        window=cars_dataset.get_window_dataset(left_disparity_object),
        profile=cars_dataset.get_profile_rasterio(left_disparity_object),
        attributes=None,
        overlaps=cars_dataset.get_overlaps_dataset(left_disparity_object),
    )

    right_pc_dataset = None
    if cst.STEREO_SEC in points:
        right_pc_dataset = points[cst.STEREO_SEC]

        cars_dataset.fill_dataset(
            right_pc_dataset,
            saving_info=saving_info_right,
            window=cars_dataset.get_window_dataset(right_disparity_object),
            profile=cars_dataset.get_profile_rasterio(right_disparity_object),
            attributes=None,
            overlaps=cars_dataset.get_overlaps_dataset(right_disparity_object),
        )

    return left_pc_dataset, right_pc_dataset
