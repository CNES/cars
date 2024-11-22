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
this module contains the statistical point removal application class.
"""


import copy

# Standard imports
import logging
import math
import time

import numpy as np

# Third party imports
from json_checker import And, Checker
from pyproj import CRS

# CARS imports
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.point_cloud_outlier_removal import outlier_removal_tools
from cars.applications.point_cloud_outlier_removal import (
    pc_out_removal as pc_removal,
)
from cars.applications.point_cloud_outlier_removal import (
    point_removal_constants as pr_cst,
)
from cars.core import projection
from cars.data_structures import cars_dataset

# R0903  temporary disabled for error "Too few public methods"
# œgoing to be corrected by adding new methods as check_conf


class Statistical(
    pc_removal.PointCloudOutlierRemoval, short_name="statistical"
):  # pylint: disable=R0903
    """
    Statistical
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf=None):
        """
        Init function of Statistical

        :param conf: configuration for points outlier removal
        :return: a application_to_use object
        """

        super().__init__(conf=conf)

        self.used_method = self.used_config["method"]

        # statistical outliers
        self.activated = self.used_config["activated"]
        self.k = self.used_config["k"]
        self.std_dev_factor = self.used_config["std_dev_factor"]
        self.use_median = self.used_config["use_median"]
        self.half_epipolar_size = self.used_config["half_epipolar_size"]

        # Saving files
        self.save_by_pair = self.used_config.get("save_by_pair", False)
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

        overloaded_conf[application_constants.SAVE_INTERMEDIATE_DATA] = (
            conf.get(application_constants.SAVE_INTERMEDIATE_DATA, False)
        )
        overloaded_conf["save_by_pair"] = conf.get("save_by_pair", False)
        overloaded_conf["use_median"] = conf.get("use_median", True)

        # statistical outlier filtering
        overloaded_conf["activated"] = conf.get(
            "activated", False
        )  # if false, the following
        # parameters are unused
        # k: number of neighbors
        overloaded_conf["k"] = conf.get("k", 50)
        # stdev_factor: factor to apply in the distance threshold computation
        overloaded_conf["std_dev_factor"] = conf.get("std_dev_factor", 5.0)

        # half_epipolar_size:
        # Half size of the epipolar window used for neighobr search (depth map
        # input only)
        overloaded_conf["half_epipolar_size"] = conf.get(
            "half_epipolar_size", 5
        )

        point_cloud_fusion_schema = {
            "method": str,
            "save_by_pair": bool,
            "activated": bool,
            "k": And(int, lambda x: x > 0),
            "std_dev_factor": And(float, lambda x: x > 0),
            "use_median": bool,
            "half_epipolar_size": int,
            application_constants.SAVE_INTERMEDIATE_DATA: bool,
        }

        # Check conf
        checker = Checker(point_cloud_fusion_schema)
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
            "Estimated optimal tile size for statistical "
            "removal: {} meters".format(tile_size)
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

        return 0

    def run(
        self,
        merged_point_cloud,
        orchestrator=None,
        output_dir=None,
        dump_dir=None,
        epsg=None,
    ):
        """
        Run PointCloudOutlierRemoval application.

        Creates a CarsDataset filled with new point cloud tiles.

        :param merged_point_cloud: merged point cloud. CarsDataset contains:

            - Z x W Delayed tiles. \
                Each tile will be a future pandas DataFrame containing:

                - data : with keys : "x", "y", "z", "corr_msk" \
                    optional: "clr", "msk", "data_valid","coord_epi_geom_i",\
                     "coord_epi_geom_j", "idx_im_epi"
                - attrs with keys: "epsg"
            - attributes containing "bounds", "ysize", "xsize", "epsg"

        :type merged_point_cloud: CarsDataset filled with pandas.DataFrame
        :param orchestrator: orchestrator used
        :param output_dir: output depth map directory. If None output will be
            written in dump_dir if intermediate data is requested
        :type output_dir: str
        :param dump_dir: dump dir for output (except depth map) if intermediate
            data is requested
        :type dump_dir: str
        :param epsg: cartographic reference for the point cloud (array input)
        :type epsg: int

        :return: filtered merged point cloud. CarsDataset contains:

            - Z x W Delayed tiles. \
                Each tile will be a future pandas DataFrame containing:

                - data : with keys "x", "y", "z", "corr_msk" \
                    optional: "clr", "msk", "data_valid", "coord_epi_geom_i",\
                     "coord_epi_geom_j", "idx_im_epi"
                - attrs with keys: "epsg"
            - attributes containing "bounds", "ysize", "xsize", "epsg"

        :rtype : CarsDataset filled with xr.Dataset
        """

        if not self.activated:
            return merged_point_cloud

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

        if merged_point_cloud.dataset_type == "points":
            (
                filtered_point_cloud,
                point_cloud_laz_file_name,
                point_cloud_csv_file_name,
            ) = self.__register_pc_dataset__(
                merged_point_cloud,
                output_dir,
                dump_dir,
                app_name="statistical",
            )

            # Get saving infos in order to save tiles when they are computed
            [saving_info] = self.orchestrator.get_saving_infos(
                [filtered_point_cloud]
            )

            # Add infos to orchestrator.out_json
            updating_dict = {
                application_constants.APPLICATION_TAG: {
                    pr_cst.CLOUD_OUTLIER_REMOVAL_RUN_TAG: {},
                }
            }
            orchestrator.update_out_info(updating_dict)
            logging.info(
                "Cloud filtering: Filtered points number: {}".format(
                    filtered_point_cloud.shape[1]
                    * filtered_point_cloud.shape[0]
                )
            )

            # Generate rasters
            for col in range(filtered_point_cloud.shape[1]):
                for row in range(filtered_point_cloud.shape[0]):
                    # update saving infos  for potential replacement
                    full_saving_info = ocht.update_saving_infos(
                        saving_info, row=row, col=col
                    )
                    if merged_point_cloud.tiles[row][col] is not None:
                        # Delayed call to cloud filtering
                        filtered_point_cloud[
                            row, col
                        ] = self.orchestrator.cluster.create_task(
                            statistical_removal_wrapper
                        )(
                            merged_point_cloud[row, col],
                            self.k,
                            self.std_dev_factor,
                            self.use_median,
                            save_by_pair=(self.save_by_pair),
                            point_cloud_csv_file_name=point_cloud_csv_file_name,
                            point_cloud_laz_file_name=point_cloud_laz_file_name,
                            saving_info=full_saving_info,
                        )
        elif merged_point_cloud.dataset_type == "arrays":
            filtered_point_cloud, saving_info = (
                self.__register_epipolar_dataset__(
                    merged_point_cloud,
                    output_dir,
                    dump_dir,
                    app_name="statistical",
                )
            )

            # Generate rasters
            for col in range(filtered_point_cloud.shape[1]):
                for row in range(filtered_point_cloud.shape[0]):

                    # update saving infos  for potential replacement
                    full_saving_info = ocht.update_saving_infos(
                        saving_info, row=row, col=col
                    )
                    if merged_point_cloud[row][col] is not None:

                        window = merged_point_cloud.tiling_grid[row, col]
                        overlap = merged_point_cloud.overlaps[row, col]
                        # Delayed call to cloud filtering
                        filtered_point_cloud[
                            row, col
                        ] = self.orchestrator.cluster.create_task(
                            epipolar_statistical_removal_wrapper
                        )(
                            merged_point_cloud[row, col],
                            self.k,
                            self.std_dev_factor,
                            self.use_median,
                            self.half_epipolar_size,
                            window,
                            overlap,
                            epsg=epsg,
                            saving_info=full_saving_info,
                        )

        else:
            logging.error(
                "PointCloudOutlierRemoval application doesn't support"
                "this input data "
                "format"
            )

        return filtered_point_cloud


def statistical_removal_wrapper(
    cloud,
    statistical_k,
    std_dev_factor,
    use_median,
    save_by_pair: bool = False,
    point_cloud_csv_file_name=None,
    point_cloud_laz_file_name=None,
    saving_info=None,
):
    """
    Statistical outlier removal

    :param cloud: cloud to filter
    :type cloud: pandas DataFrame
    :param statistical_k: k
    :type statistical_k: int
    :param std_dev_factor: std factor
    :type std_dev_factor: float
    :param use_median: use median and quartile instead of mean and std
    :type use median: bool
    :param save_by_pair: save point cloud as pair
    :type save_by_pair: bool
    :param point_cloud_csv_file_name: write point cloud as CSV in filename
        (if None, the point cloud is not written as csv)
    :type point_cloud_csv_file_name: str
    :param point_cloud_laz_file_name: write point cloud as laz in filename
        (if None, the point cloud is not written as laz)
    :type point_cloud_laz_file_name: str
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
            "The point cloud to filter is not in a cartographic system. "
            "The filter's default parameters might not be adapted "
            "to this referential. Convert the points "
            "cloud to ECEF to ensure a proper point_cloud."
        )
        # Convert to epsg = 4978
        cartographic_epsg = 4978
        projection.point_cloud_conversion_dataframe(
            new_cloud, current_epsg, cartographic_epsg
        )
        current_epsg = cartographic_epsg

    # Filter point cloud
    tic = time.process_time()
    (new_cloud, _) = outlier_removal_tools.statistical_outlier_filtering(
        new_cloud, statistical_k, std_dev_factor, use_median
    )
    toc = time.process_time()
    logging.debug(
        "Statistical cloud filtering done in {} seconds".format(toc - tic)
    )

    # Conversion to UTM
    projection.point_cloud_conversion_dataframe(
        new_cloud, cloud_epsg, current_epsg
    )
    # Update attributes
    cloud_attributes["epsg"] = current_epsg

    cars_dataset.fill_dataframe(
        new_cloud, saving_info=saving_info, attributes=cloud_attributes
    )

    # save point cloud in worker
    if point_cloud_csv_file_name:
        cars_dataset.run_save_points(
            new_cloud,
            point_cloud_csv_file_name,
            save_by_pair=save_by_pair,
            overwrite=True,
            point_cloud_format="csv",
        )
    if point_cloud_laz_file_name:
        cars_dataset.run_save_points(
            new_cloud,
            point_cloud_laz_file_name,
            save_by_pair=save_by_pair,
            overwrite=True,
            point_cloud_format="laz",
        )

    return new_cloud


def epipolar_statistical_removal_wrapper(
    epipolar_ds,
    statistical_k,
    std_dev_factor,
    use_median,
    half_epipolar_size,
    window,
    overlap,
    epsg,
    saving_info=None,
):
    """
    Statistical outlier removal in epipolar geometry

    :param epipolar_ds: epipolar dataset to filter
    :type epipolar_ds: xr.Dataset
    :param statistical_k: k
    :type statistical_k: int
    :param std_dev_factor: std factor
    :type std_dev_factor: float
    :param use_median: use median and quartile instead of mean and std
    :type use median: bool
    :param half_epipolar_size: half size of the window used to search neighbors
    :type half_epipolar_size: int
    :param window: window of base tile [row min, row max, col min col max]
    :type window: list
    :param overlap: overlap [row min, row max, col min col max]
    :type overlap: list
    :param epsg: epsg code of the CRS used to compute distances
    :type epsg: int

    :return: filtered dataset
    :rtype:  xr.Dataset

    """

    # Copy input cloud
    filtered_cloud = copy.copy(epipolar_ds)

    outlier_removal_tools.epipolar_statistical_filtering(
        filtered_cloud,
        epsg,
        k=statistical_k,
        dev_factor=std_dev_factor,
        use_median=use_median,
        half_window_size=half_epipolar_size,
    )

    # Fill with attributes
    cars_dataset.fill_dataset(
        filtered_cloud,
        saving_info=saving_info,
        window=cars_dataset.window_array_to_dict(window),
        profile=None,
        attributes=None,
        overlaps=cars_dataset.overlap_array_to_dict(overlap),
    )

    return filtered_cloud
