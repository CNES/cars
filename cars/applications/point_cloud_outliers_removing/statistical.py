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
from json_checker import Checker
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

# R0903  temporary disabled for error "Too few public methods"
# œgoing to be corrected by adding new methods as check_conf


class Statistical(
    pc_removing.PointCloudOutliersRemoving, short_name="statistical"
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

        # statistical outliers
        self.activated = checked_conf["activated"]
        self.k = checked_conf["k"]
        self.std_dev_factor = checked_conf["std_dev_factor"]

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
        overloaded_conf["method"] = conf.get("method", "statistical")
        overloaded_conf["save_points_cloud"] = conf.get(
            "save_points_cloud", False
        )

        # statistical outlier filtering
        overloaded_conf["activated"] = conf.get(
            "activated", True
        )  # if false, the following
        # parameters are unused
        # k: number of neighbors
        overloaded_conf["k"] = conf.get("k", 50)
        # stdev_factor: factor to apply in the distance threshold computation
        overloaded_conf["std_dev_factor"] = conf.get("std_dev_factor", 5.0)

        points_cloud_fusion_schema = {
            "method": str,
            "save_points_cloud": bool,
            "activated": bool,
            "k": int,
            "std_dev_factor": float,
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

                - data : with keys : "x", "y", "z", "corr_msk" \
                    optional: "clr", "msk", "data_valid","coord_epi_geom_i",\
                     "coord_epi_geom_j", "idx_im_epi"
                - attrs with keys: "epsg"
            - attributes containing "bounds", "ysize", "xsize", "epsg"

        :type merged_points_cloud: CarsDataset filled with pandas.DataFrame
        :param orchestrator: orchestrator used

        :return: filtered merged points cloud. CarsDataset contains:

            - Z x W Delayed tiles. \
                Each tile will be a future pandas DataFrame containing:

                - data : with keys "x", "y", "z", "corr_msk" \
                    optional: "clr", "msk", "data_valid", "coord_epi_geom_i",\
                     "coord_epi_geom_j", "idx_im_epi"
                - attrs with keys: "epsg"
            - attributes containing "bounds", "ysize", "xsize", "epsg"

        :rtype : CarsDataset filled with xr.Dataset
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
                    "points_cloud_post_statistical_removing.csv",
                )
                self.orchestrator.add_to_save_lists(
                    pc_file_name,
                    None,
                    filtered_point_cloud,
                    cars_ds_name="filtered_merged_pc_statistical",
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
                        pr_cst.STATISTICAL_OUTLIER: (self.activated),
                        pr_cst.SO_K: self.k,
                        pr_cst.SO_STD_DEV_FACTOR: (self.std_dev_factor),
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
                            statistical_removing_wrapper
                        )(
                            merged_points_cloud[row, col],
                            self.activated,
                            self.k,
                            self.std_dev_factor,
                            saving_info=saving_info,
                        )

        else:
            logging.error(
                "PointCloudOutliersRemoving application doesn't support"
                "this input data "
                "format"
            )

        return filtered_point_cloud


def statistical_removing_wrapper(
    cloud, activated, statistical_k, std_dev_factor, saving_info=None
):
    """
    Statistical outlier removing

    :param cloud: cloud to filter
    :type cloud: pandas DataFrame
    :param activated: true if filtering must be done
    :type activated: bool
    :param statistical_k: k
    :type statistical_k: float
    :param std_dev_factor: std factor
    :type std_dev_factor: float
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
        (new_cloud, _) = outlier_removing_tools.statistical_outliers_filtering(
            new_cloud, statistical_k, std_dev_factor
        )
        toc = time.process_time()
        worker_logger.debug(
            "Statistical cloud filtering done in {} seconds".format(toc - tic)
        )

    # Update attributes
    cloud_attributes["epsg"] = current_epsg
    cars_dataset.fill_dataframe(
        new_cloud, saving_info=saving_info, attributes=cloud_attributes
    )

    return new_cloud
