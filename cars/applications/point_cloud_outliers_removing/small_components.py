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


# Standard imports
import logging
import os
import time

# Third party imports
from json_checker import Checker, Or
from osgeo import osr

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
        :return: a application_to_use object
        """

        # Check conf
        checked_conf = self.check_conf(conf)
        # used_config used for printing config
        self.used_config = checked_conf

        self.used_method = checked_conf["method"]

        # small components
        self.activated = checked_conf["activated"]
        self.on_ground_margin = checked_conf["on_ground_margin"]
        self.connection_distance = checked_conf["connection_distance"]
        self.nb_points_threshold = checked_conf["nb_points_threshold"]
        self.clusters_distance_threshold = checked_conf[
            "clusters_distance_threshold"
        ]

        # check loader

        # Saving files
        self.save_points_cloud = checked_conf.get("save_points_cloud", False)

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
        overloaded_conf["save_points_cloud"] = conf.get(
            "save_points_cloud", False
        )

        # small components
        overloaded_conf["activated"] = conf.get(
            "activated", True
        )  # if false, the following
        # parameters are unused
        # on_ground_margin:
        #           margin added to the on ground region to not filter
        #           points clusters
        #           that were incomplete because they were on the edges
        overloaded_conf["on_ground_margin"] = conf.get("on_ground_margin", 10)
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
            "save_points_cloud": bool,
            "activated": bool,
            "on_ground_margin": int,
            "connection_distance": float,
            "nb_points_threshold": int,
            "clusters_distance_threshold": Or(None, float),
        }

        # Check conf
        checker = Checker(points_cloud_fusion_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def get_method(self):
        """
        Get margins to use during point clouds fusion

        :return: algorithm method
        :rtype: string

        """

        return self.used_method

    def get_on_ground_margin(self):
        """
        Get margins to use during point clouds fusion

        :return: margin
        :rtype: float

        """

        on_ground_margin = 0

        if self.activated:
            on_ground_margin = self.on_ground_margin

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

            # Create CarsDataset
            filtered_point_cloud = cars_dataset.CarsDataset("points")

            # Get tiling grid
            filtered_point_cloud.tiling_grid = merged_points_cloud.tiling_grid
            filtered_point_cloud.generate_none_tiles()

            filtered_point_cloud.attributes = (
                merged_points_cloud.attributes.copy()
            )

            # Save objects
            if self.save_points_cloud:
                # Points cloud file name
                # TODO in input conf file
                pc_file_name = os.path.join(
                    self.orchestrator.out_dir,
                    "points_cloud_post_small_components_removing.csv",
                )
                self.orchestrator.add_to_save_lists(
                    pc_file_name,
                    None,
                    filtered_point_cloud,
                    cars_ds_name="filtered_merged_pc_small_components",
                )

            # Get saving infos in order to save tiles when they are computed
            [saving_info] = self.orchestrator.get_saving_infos(
                [filtered_point_cloud]
            )

            # Add infos to orchestrator.out_json
            updating_dict = {
                application_constants.APPLICATION_TAG: {
                    pr_cst.CLOUD_OUTLIER_REMOVING_PARAMS_TAG: {
                        pr_cst.METHOD: self.used_method,
                        pr_cst.SMALL_COMPONENTS_FILTER: (self.activated),
                        pr_cst.SC_ON_GROUND_MARGIN: (self.on_ground_margin),
                        pr_cst.SC_CONNECTION_DISTANCE: (
                            self.connection_distance
                        ),
                        pr_cst.SC_NB_POINTS_THRESHOLD: (
                            self.nb_points_threshold
                        ),
                        pr_cst.SC_CLUSTERS_DISTANCES_THRESHOLD: (
                            self.clusters_distance_threshold
                        ),
                    },
                    pr_cst.CLOUD_OUTLIER_REMOVING_RUN_TAG: {},
                }
            }
            orchestrator.update_out_info(updating_dict)

            # Generate rasters
            for col in range(filtered_point_cloud.shape[1]):
                for row in range(filtered_point_cloud.shape[0]):

                    if merged_points_cloud.tiles[row][col] is not None:

                        # Delayed call to cloud filtering
                        filtered_point_cloud[
                            row, col
                        ] = self.orchestrator.cluster.create_task(
                            small_components_removing_wrapper
                        )(
                            merged_points_cloud[row, col],
                            self.activated,
                            self.connection_distance,
                            self.nb_points_threshold,
                            self.clusters_distance_threshold,
                            saving_info=saving_info,
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
    activated,
    connection_distance,
    nb_points_threshold,
    clusters_distance_threshold,
    saving_info=None,
):
    """
    Statistical outlier removing

    :param cloud: cloud to filter
    :type cloud: pandas DataFrame
    :param activated: true if filtering must be done
    :type activated: bool
    :param connection_distance: connection distance
    :type connection_distance: float
    :param nb_points_threshold:
    :type nb_points_threshold: int
    :param clusters_distance_threshold:
    :type clusters_distance_threshold: float
    :param saving_info: saving infos
    :type saving_info: dict

    :return: filtered cloud
    :rtype: pandas DataFrame

    """

    # Copy input cloud
    new_cloud = cloud.copy()
    new_cloud.attrs = cloud.attrs.copy()

    if activated:
        worker_logger = logging.getLogger("distributed.worker")
        # Get current epsg
        cloud_attributes = cars_dataset.get_attributes_dataframe(new_cloud)
        cloud_epsg = cloud_attributes["epsg"]
        current_epsg = cloud_epsg

        # Check if can be used to filter
        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromEPSG(cloud_epsg)
        if spatial_ref.IsGeographic():

            worker_logger.debug(
                "The points cloud to filter is not in a cartographic system. "
                "The filter's default parameters might not be adapted "
                "to this referential. Convert the points "
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
        (new_cloud, _,) = outlier_removing_tools.small_components_filtering(
            new_cloud,
            connection_distance,
            nb_points_threshold,
            clusters_distance_threshold,
        )
        toc = time.process_time()
        worker_logger.debug(
            "Small components cloud filtering done in {} seconds".format(
                toc - tic
            )
        )

    # Update attributes
    cloud_attributes["epsg"] = current_epsg
    cars_dataset.fill_dataframe(
        new_cloud, saving_info=saving_info, attributes=cloud_attributes
    )

    return new_cloud
