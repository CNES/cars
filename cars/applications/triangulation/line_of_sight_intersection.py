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
from cars.core import inputs, projection, tiling

# CARS imports
from cars.core.geometry.abstract_geometry import read_geoid_file
from cars.core.utils import safe_makedirs
from cars.data_structures import cars_dataset
from cars.pipelines.sensor_to_dense_dsm import (
    sensor_dense_dsm_constants as sens_cst,
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

        # Saving files
        overloaded_conf["save_points_cloud"] = conf.get(
            "save_points_cloud", False
        )

        triangulation_schema = {
            "method": str,
            "use_geoid_alt": bool,
            "snap_to_img1": bool,
            "add_msk_info": bool,
            "save_points_cloud": bool,
        }

        # Check conf
        checker = Checker(triangulation_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def run(  # noqa: C901
        self,
        sensor_image_left,
        sensor_image_right,
        epipolar_image,
        grid_left,
        grid_right,
        epipolar_disparity_map,
        epsg,
        geometry_plugin,
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
        :param epipolar_image: tiled epipolar left image
        :type epipolar_image: CarsDataset
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
        :param epipolar_disparity_map: tiled left disparity map or \
            sparse matches:

            - if CarsDataset is instance of "arrays", CarsDataset contains:

                - N x M Delayed tiles \
                    Each tile will be a future xarray Dataset containing:

                    - data with keys : "disp", "disp_msk"
                    - attrs with keys: profile, window, overlaps
                - attributes containing:"largest_epipolar_region"\
                  "opt_epipolar_tile_size"

            - if CarsDataset is instance of "points", CarsDataset contains:

                - N x M Delayed tiles \
                    Each tile will be a future pandas DataFrame containing:

                    - data : (L, 4) shape matches
                - attributes containing:"disp_lower_bound","disp_upper_bound",\
                    "elevation_delta_lower_bound","elevation_delta_upper_bound"

        :type epipolar_disparity_map: CarsDataset
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

        :return: points cloud \
                The CarsDataset contains:

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
                    "have different images as their "
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

        sensor1 = sensor_image_left[sens_cst.INPUT_IMG]
        sensor2 = sensor_image_right[sens_cst.INPUT_IMG]
        geomodel1 = sensor_image_left[sens_cst.INPUT_GEO_MODEL]
        geomodel2 = sensor_image_right[sens_cst.INPUT_GEO_MODEL]

        if self.snap_to_img1:
            grid_right = uncorrected_grid_right
            if grid_right is None:
                logging.error(
                    "Uncorrected grid was not given in order to snap it to img1"
                )

        # Compute disp_min and disp_max location for epipolar grid
        (
            epipolar_grid_min,
            epipolar_grid_max,
        ) = grids.compute_epipolar_grid_min_max(
            geometry_plugin,
            tiling.transform_four_layers_to_two_layers_grid(
                epipolar_image.tiling_grid
            ),
            sensor1,
            sensor2,
            geomodel1,
            geomodel2,
            grid_left,
            grid_right,
            epsg,
            disp_min,
            disp_max,
        )
        # update attributes for corresponding tiles in cloud fusion
        # TODO remove with refactoring
        pc_attributes = {
            "used_epsg_for_terrain_grid": epsg,
            "epipolar_grid_min": epipolar_grid_min,
            "epipolar_grid_max": epipolar_grid_max,
            "largest_epipolar_region": epipolar_image.attributes[
                "largest_epipolar_region"
            ],
            "opt_epipolar_tile_size": epipolar_image.attributes[
                "opt_epipolar_tile_size"
            ],
        }

        if epipolar_disparity_map.dataset_type in ("arrays", "points"):
            # Create CarsDataset
            # Epipolar_point_cloud
            epipolar_points_cloud = cars_dataset.CarsDataset(
                epipolar_disparity_map.dataset_type
            )
            epipolar_points_cloud.create_empty_copy(epipolar_image)
            epipolar_points_cloud.overlaps *= 0  # Margins removed

            # Update attributes to get epipolar info
            epipolar_points_cloud.attributes.update(pc_attributes)

            # Save objects
            if self.save_points_cloud:
                # if isinstance(epipolar_points_cloud, xr.DataArray):
                if epipolar_disparity_map.dataset_type == "arrays":
                    # Propagate color type in output file
                    color_type = None
                    if sens_cst.INPUT_COLOR in sensor_image_left:
                        color_type = inputs.rasterio_get_image_type(
                            sensor_image_left[sens_cst.INPUT_COLOR]
                        )
                    else:
                        color_type = inputs.rasterio_get_image_type(
                            sensor_image_left[sens_cst.INPUT_IMG]
                        )

                    self.orchestrator.add_to_save_lists(
                        os.path.join(pair_folder, "epi_pc_X.tif"),
                        cst.X,
                        epipolar_points_cloud,
                        cars_ds_name="epi_pc_x",
                    )

                    self.orchestrator.add_to_save_lists(
                        os.path.join(pair_folder, "epi_pc_Y.tif"),
                        cst.Y,
                        epipolar_points_cloud,
                        cars_ds_name="epi_pc_y",
                    )

                    self.orchestrator.add_to_save_lists(
                        os.path.join(pair_folder, "epi_pc_Z.tif"),
                        cst.Z,
                        epipolar_points_cloud,
                        cars_ds_name="epi_pc_z",
                    )

                    self.orchestrator.add_to_save_lists(
                        os.path.join(pair_folder, "epi_classification.tif"),
                        cst.EPI_CLASSIFICATION,
                        epipolar_points_cloud,
                        cars_ds_name="epi_classification",
                        optional_data=True,
                    )

                    self.orchestrator.add_to_save_lists(
                        os.path.join(pair_folder, "epi_filling.tif"),
                        cst.EPI_FILLING,
                        epipolar_points_cloud,
                        cars_ds_name="epi_filling",
                        optional_data=True,
                    )

                    self.orchestrator.add_to_save_lists(
                        os.path.join(pair_folder, "epi_pc_color.tif"),
                        cst.EPI_COLOR,
                        epipolar_points_cloud,
                        cars_ds_name="epi_pc_color",
                        dtype=color_type,
                    )
                else:
                    self.orchestrator.add_to_save_lists(
                        os.path.join(pair_folder, "epi_pc"),
                        cst.POINTS_CLOUD_MATCHES,
                        epipolar_points_cloud,
                        cars_ds_name="epi_pc_x",
                    )

        else:
            logging.error(
                "Triangulation application doesn't "
                "support this input data format"
            )

        # Get saving infos in order to save tiles when they are computed
        [saving_info] = self.orchestrator.get_saving_infos(
            [epipolar_points_cloud]
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

        for col in range(epipolar_image.shape[1]):
            for row in range(epipolar_image.shape[0]):
                if type(None) not in (
                    type(epipolar_disparity_map[row, col]),
                    type(epipolar_image[row, col]),
                ):
                    # update saving infos  for potential replacement
                    full_saving_info = ocht.update_saving_infos(
                        saving_info, row=row, col=col
                    )
                    # Compute points
                    (
                        epipolar_points_cloud[row][col]
                    ) = self.orchestrator.cluster.create_task(
                        compute_points_cloud
                    )(
                        epipolar_image[row, col],
                        epipolar_disparity_map[row, col],
                        sensor1,
                        sensor2,
                        geomodel1,
                        geomodel2,
                        grid_left,
                        grid_right,
                        geometry_plugin,
                        epsg,
                        geoid_data=geoid_data_futures,
                        add_msk_info=self.add_msk_info,
                        saving_info=full_saving_info,
                    )

        return epipolar_points_cloud


def compute_points_cloud(
    image_object: xr.Dataset,
    disparity_object: xr.Dataset,
    sensor1,
    sensor2,
    geomodel1,
    geomodel2,
    grid1,
    grid2,
    geometry_plugin,
    epsg,
    geoid_data: xr.Dataset = None,
    add_msk_info: bool = False,
    saving_info=None,
) -> Dict[str, Tuple[xr.Dataset, xr.Dataset]]:
    """
    Compute points clouds from image objects and disparity objects.

    :param image_object: Left image dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_COLOR (for left, if given)
    :type image_object: xr.Dataset with :

            - cst.EPI_IMAGE
            - cst.EPI_MSK (if given)
            - cst.EPI_COLOR (for left, if given)
    :param disparity_object: Left disparity map dataset with :

            - cst_disp.MAP
            - cst_disp.VALID
            - cst.EPI_COLOR
    :type disparity_object: xr.Dataset
    :param sensor1: path to left sensor image
    :type sensor1: str
    :param sensor2: path to right sensor image
    :type sensor2: str
    :param geomodel1: path and attributes for left geomodel
    :type geomodel1: dict
    :param geomodel2: path and attributes for right geomodel
    :type geomodel2: dict
    :param grid1: dataset of the reference image grid file
    :type grid1: CarsDataset
    :param grid2: dataset of the secondary image grid file
    :type grid2: CarsDataset
    :param geometry_plugin: geometry plugin to use
    :type geometry_plugin: AbstractGeometry
    :param geoid_data: Geoid used for altimetric reference. Defaults to None
        for using ellipsoid as altimetric reference.
    :type geoid_data: str
    :param snap_to_img1: If True, Lines of Sight of img2 are moved so as to
                         cross those of img1
    :type snap_to_img1: bool
    :param add_msk_info:  boolean enabling the addition of the masks'
                         information in the point clouds final dataset
    :type add_msk_info: bool

    :return: Left disparity object

    Returned object is composed of :

        - dataset with :

            - cst.X
            - cst.Y
            - cst.Z
            - cst.EPI_COLOR
    """
    # Get disparity maps
    disp_ref = disparity_object

    # Get masks
    left = image_object
    im_ref_msk = None
    if add_msk_info:
        ref_values_list = [key for key, _ in left.items()]
        if cst.EPI_MSK in ref_values_list:
            im_ref_msk = left
        else:
            worker_logger = logging.getLogger("distributed.worker")
            worker_logger.warning(
                "Left image does not have a mask to rasterize"
            )

    # Triangulate
    if isinstance(disp_ref, xr.Dataset):
        # Triangulate epipolar dense disparities
        points = triangulation_tools.triangulate(
            geometry_plugin,
            sensor1,
            sensor2,
            geomodel1,
            geomodel2,
            grid1,
            grid2,
            disp_ref,
            im_ref_msk_ds=im_ref_msk,
        )
    elif isinstance(disp_ref, pandas.DataFrame):
        # Triangulate epipolar sparse matches
        points = {}
        points[cst.STEREO_REF] = triangulation_tools.triangulate_matches(
            geometry_plugin,
            sensor1,
            sensor2,
            geomodel1,
            geomodel2,
            grid1,
            grid2,
            disp_ref.to_numpy(),
        )
    else:
        logging.error(
            "Disp ref is neither xarray Dataset  nor pandas DataFrame"
        )
        raise TypeError(
            "Disp ref is neither xarray Dataset  nor pandas DataFrame"
        )

    if geoid_data is not None:  # if user pass a geoid, use it a alt reference
        for key, point in points.items():
            points[key] = triangulation_tools.geoid_offset(point, geoid_data)

    # propagate the color type
    color_type = None
    if cst.EPI_COLOR in image_object.data_vars.keys():
        color_type = image_object[cst.EPI_COLOR].attrs["color_type"]
    else:
        color_type = image_object[cst.EPI_IMAGE].attrs["color_type"]

    # Fill datasets
    pc_dataset = points[cst.STEREO_REF]

    if color_type:
        pc_dataset.attrs["color_type"] = color_type
    attributes = None
    if isinstance(disp_ref, pandas.DataFrame):
        # Conversion to UTM
        projection.points_cloud_conversion_dataframe(
            points[cst.STEREO_REF], points[cst.STEREO_REF].attrs[cst.EPSG], epsg
        )
        cloud_epsg = epsg
        pc_dataset.attrs["epsg"] = cloud_epsg
        attributes = {
            "save_points_cloud_as_laz": True,
            "epsg": cloud_epsg,
            "color_type": None,
        }
        cars_dataset.fill_dataframe(
            pc_dataset, saving_info=saving_info, attributes=attributes
        )
    else:
        cars_dataset.fill_dataset(
            pc_dataset,
            saving_info=saving_info,
            window=cars_dataset.get_window_dataset(disparity_object),
            profile=cars_dataset.get_profile_rasterio(disparity_object),
            attributes=attributes,
            overlaps=cars_dataset.get_overlaps_dataset(disparity_object),
        )

    return pc_dataset
