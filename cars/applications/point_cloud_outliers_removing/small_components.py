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
this module contains the statistical points removing application class.
"""


import copy

# Standard imports
import logging
import math
import time

# Third party imports
import numpy as np
from json_checker import And, Checker, Or
from pyproj import CRS

# CARS imports
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.point_cloud_outliers_removing import (
    outlier_removing_tools,
)
from cars.applications.point_cloud_outliers_removing import (
    pc_out_removing as pc_removing,
)
from cars.applications.point_cloud_outliers_removing import (
    points_removing_constants as pr_cst,
)
from cars.core import projection
from cars.data_structures import cars_dataset


class SmallComponents(
    pc_removing.PointCloudOutliersRemoving, short_name="small_components"
):  # pylint: disable=R0903
    """
    PointCloudOutliersRemoving
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf=None):
        """
        Init function of PointCloudOutliersRemoving

        :param conf: configuration for points outliers removing
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

        self.used_method = self.used_config["method"]

        # small components
        self.activated = self.used_config["activated"]
        self.on_ground_margin = self.used_config["on_ground_margin"]
        self.connection_distance = self.used_config["connection_distance"]
        self.nb_points_threshold = self.used_config["nb_points_threshold"]
        self.clusters_distance_threshold = self.used_config[
            "clusters_distance_threshold"
        ]

        # Saving files
        self.save_points_cloud_as_laz = self.used_config.get(
            "save_points_cloud_as_laz", False
        )
        self.save_points_cloud_as_csv = self.used_config.get(
            "save_points_cloud_as_csv", False
        )
        self.save_points_cloud_by_pair = self.used_config.get(
            "save_points_cloud_by_pair", False
        )

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
        overloaded_conf["method"] = conf.get("method", "small_components")
        overloaded_conf["save_points_cloud_as_laz"] = conf.get(
            "save_points_cloud_as_laz", False
        )
        overloaded_conf["save_points_cloud_as_csv"] = conf.get(
            "save_points_cloud_as_csv", False
        )
        overloaded_conf["save_points_cloud_by_pair"] = conf.get(
            "save_points_cloud_by_pair", False
        )

        # small components
        overloaded_conf["activated"] = conf.get(
            "activated", False
        )  # if false, the following
        # parameters are unused
        # on_ground_margin:
        #           margin added to the on ground region to not filter
        #           points clusters
        #           that were incomplete because they were on the edges
        overloaded_conf["on_ground_margin"] = conf.get("on_ground_margin", 11)
        # pts_connection_dist:
        #           distance to use to consider that two points are connected
        overloaded_conf["connection_distance"] = conf.get(
            "connection_distance", 3.0
        )
        # nb_pts_threshold:
        #           points clusters that have less than this number of points
        #           will be filtered
        overloaded_conf["nb_points_threshold"] = conf.get(
            "nb_points_threshold", 50
        )
        # dist_between_clusters:
        #           distance to use to consider that two points clusters
        #           are far from each other or not.
        #       If a small points cluster is near to another one, it won't
        #           be filtered.
        #          (None = deactivated)
        overloaded_conf["clusters_distance_threshold"] = conf.get(
            "clusters_distance_threshold", None
        )

        points_cloud_fusion_schema = {
            "method": str,
            "save_points_cloud_as_laz": bool,
            "save_points_cloud_as_csv": bool,
            "save_points_cloud_by_pair": bool,
            "activated": bool,
            "on_ground_margin": int,
            "connection_distance": And(float, lambda x: x > 0),
            "nb_points_threshold": And(int, lambda x: x > 0),
            "clusters_distance_threshold": Or(None, float),
        }

        # Check conf
        checker = Checker(points_cloud_fusion_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def get_optimal_tile_size(
        self,
        max_ram_per_worker,
        superposing_point_clouds=1,
        point_cloud_resolution=0.5,
    ):
        """
        Get the optimal tile size to use, depending on memory available

        :param max_ram_per_worker: maximum ram available
        :type max_ram_per_worker: int
        :param superposing_point_clouds: number of point clouds superposing
        :type superposing_point_clouds: int
        :param point_cloud_resolution: resolution of point cloud
        :type point_cloud_resolution: float

        :return: optimal tile size in meter
        :rtype: float

        """

        if not self.activated:
            # if not activated, this tile size must not be taken into acount
            # during the min(*tile_sizes) operations
            tile_size = math.inf
        else:
            tot = 10000 * superposing_point_clouds / point_cloud_resolution

            import_ = 200  # MiB
            tile_size = int(
                np.sqrt(float(((max_ram_per_worker - import_) * 2**23)) / tot)
            )

        logging.info(
            "Estimated optimal tile size for small"
            "components removing: {} meters".format(tile_size)
        )

        return tile_size

    def get_method(self):
        """
        Get margins to use during point clouds fusion

        :return: algorithm method
        :rtype: string

        """

        return self.used_method

    def get_on_ground_margin(self, resolution=0.5):
        """
        Get margins to use during point clouds fusion

        :return: margin
        :rtype: float

        """

        on_ground_margin = 0

        if self.activated:
            on_ground_margin = self.on_ground_margin * resolution

        return on_ground_margin

    def run(
        self,
        merged_points_cloud,
        orchestrator=None,
    ):
        """
        Run PointCloudOutliersRemoving application.

        Creates a CarsDataset filled with new point cloud tiles.

        :param merged_points_cloud: merged point cloud. CarsDataset contains:

            - Z x W Delayed tiles. \
                Each tile will be a future pandas DataFrame containing:

                - data : with keys : "x", "y", "z", "corr_msk"\
                    optional: "clr", "msk", "data_valid","coord_epi_geom_i",\
                     "coord_epi_geom_j","idx_im_epi" \
                - attrs with keys: "epsg"
            - attributes containing "bounds", "ysize", "xsize", "epsg"

        :type merged_points_cloud: CarsDataset filled with pandas.DataFrame
        :param orchestrator: orchestrator used

        :return: filtered merged points cloud. CarsDataset contains:

            - Z x W Delayed tiles.\
                Each tile will be a future pandas DataFrame containing:

                - data : with keys : "x", "y", "z", "corr_msk"\
                    optional: "clr", "msk", "data_valid","coord_epi_geom_i",\
                     "coord_epi_geom_j","idx_im_epi"
                - attrs with keys: "epsg"
            - attributes containing "bounds", "ysize", "xsize", "epsg"

        :rtype: CarsDataset filled with xr.Dataset
        """

        if not self.activated:
            return merged_points_cloud

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

        if merged_points_cloud.dataset_type == "points":
            (filtered_point_cloud, pc_file_name) = self.__register_dataset__(
                merged_points_cloud,
                self.save_points_cloud_as_laz,
                self.save_points_cloud_as_csv,
                app_name="small_components",
            )

            # Get saving infos in order to save tiles when they are computed
            [saving_info] = self.orchestrator.get_saving_infos(
                [filtered_point_cloud]
            )

            # Add infos to orchestrator.out_json
            updating_dict = {
                application_constants.APPLICATION_TAG: {
                    pr_cst.CLOUD_OUTLIER_REMOVING_RUN_TAG: {},
                }
            }
            orchestrator.update_out_info(updating_dict)

            # Generate rasters
            for col in range(filtered_point_cloud.shape[1]):
                for row in range(filtered_point_cloud.shape[0]):
                    # update saving infos  for potential replacement
                    full_saving_info = ocht.update_saving_infos(
                        saving_info, row=row, col=col
                    )
                    if merged_points_cloud.tiles[row][col] is not None:
                        # Delayed call to cloud filtering
                        filtered_point_cloud[
                            row, col
                        ] = self.orchestrator.cluster.create_task(
                            small_components_removing_wrapper
                        )(
                            merged_points_cloud[row, col],
                            self.connection_distance,
                            self.nb_points_threshold,
                            self.clusters_distance_threshold,
                            self.save_points_cloud_as_laz,
                            self.save_points_cloud_as_csv,
                            save_points_cloud_by_pair=(
                                self.save_points_cloud_by_pair
                            ),
                            point_cloud_file_name=pc_file_name,
                            saving_info=full_saving_info,
                        )

        else:
            logging.error(
                "PointCloudOutliersRemoving application doesn't support"
                "this input data "
                "format"
            )

        return filtered_point_cloud


def small_components_removing_wrapper(
    cloud,
    connection_distance,
    nb_points_threshold,
    clusters_distance_threshold,
    save_points_cloud_as_laz,
    save_points_cloud_as_csv,
    save_points_cloud_by_pair: bool = False,
    point_cloud_file_name=None,
    saving_info=None,
):
    """
    Statistical outlier removing

    :param cloud: cloud to filter
    :type cloud: pandas DataFrame
    :param connection_distance: connection distance
    :type connection_distance: float
    :param nb_points_threshold:
    :type nb_points_threshold: int
    :param clusters_distance_threshold:
    :type clusters_distance_threshold: float
    :param save_points_cloud_as_laz: activation of point cloud saving to laz
    :type save_points_cloud_as_laz: bool
    :param save_points_cloud_as_csv: activation of point cloud saving to csv
    :type save_points_cloud_as_csv: bool
    :param save_points_cloud_by_pair: save point cloud as pair
    :type save_points_cloud_by_pair: bool
    :param point_cloud_file_name: point cloud filename
    :type point_cloud_file_name: str
    :param saving_info: saving infos
    :type saving_info: dict

    :return: filtered cloud
    :rtype: pandas DataFrame

    """

    # Copy input cloud
    new_cloud = cloud.copy()
    new_cloud.attrs = copy.deepcopy(cloud.attrs)

    # Get current epsg
    cloud_attributes = cars_dataset.get_attributes_dataframe(new_cloud)
    cloud_epsg = cloud_attributes["epsg"]
    current_epsg = cloud_epsg

    # Check if can be used to filter
    spatial_ref = CRS.from_epsg(cloud_epsg)
    if spatial_ref.is_geographic:
        logging.debug(
            "The points cloud to filter is not in a cartographic system. "
            "The filter's default parameters might not be adapted "
            "to this referential. Please, convert the points "
            "cloud to ECEF to ensure a proper points_cloud."
        )
        # Convert to epsg = 4978
        cartographic_epsg = 4978

        projection.points_cloud_conversion_dataframe(
            new_cloud, current_epsg, cartographic_epsg
        )
        current_epsg = cartographic_epsg

    # Filter point cloud
    tic = time.process_time()
    (
        new_cloud,
        _,
    ) = outlier_removing_tools.small_components_filtering(
        new_cloud,
        connection_distance,
        nb_points_threshold,
        clusters_distance_threshold,
    )
    toc = time.process_time()
    logging.debug(
        "Small components cloud filtering done in {} seconds".format(toc - tic)
    )

    # Conversion to UTM
    projection.points_cloud_conversion_dataframe(
        new_cloud, cloud_epsg, current_epsg
    )
    # Update attributes
    cloud_attributes["epsg"] = current_epsg
    cloud_attributes["save_points_cloud_as_laz"] = save_points_cloud_as_laz
    cloud_attributes["save_points_cloud_as_csv"] = save_points_cloud_as_csv
    cars_dataset.fill_dataframe(
        new_cloud, saving_info=saving_info, attributes=cloud_attributes
    )

    # save point cloud in worker
    if save_points_cloud_as_laz or save_points_cloud_as_csv:
        cars_dataset.run_save_points(
            new_cloud,
            point_cloud_file_name,
            save_points_cloud_by_pair=save_points_cloud_by_pair,
            overwrite=True,
        )

    return new_cloud
