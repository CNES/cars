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
this module contains the small_components point removal application class.
"""


import copy

# Standard imports
import logging
import math
import os
import time

# Third party imports
import numpy as np
from json_checker import And, Checker, Or
from pyproj import CRS

# CARS imports
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.point_cloud_fusion import point_cloud_tools
from cars.applications.point_cloud_outlier_removal import outlier_removal_tools
from cars.applications.point_cloud_outlier_removal import (
    pc_out_removal as pc_removal,
)
from cars.applications.triangulation.triangulation_tools import (
    generate_point_cloud_file_names,
)
from cars.core import constants as cst
from cars.core import projection
from cars.data_structures import cars_dataset


class SmallComponents(
    pc_removal.PointCloudOutlierRemoval, short_name="small_components"
):  # pylint: disable=R0903
    """
    SmallComponents
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, conf=None):
        """
        Init function of SmallComponents

        :param conf: configuration for points outlier removal
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
        overloaded_conf["method"] = conf.get("method", "small_components")
        overloaded_conf[application_constants.SAVE_INTERMEDIATE_DATA] = (
            conf.get(application_constants.SAVE_INTERMEDIATE_DATA, False)
        )
        overloaded_conf["save_by_pair"] = conf.get("save_by_pair", False)

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
            "on_ground_margin": int,
            "connection_distance": And(float, lambda x: x > 0),
            "nb_points_threshold": And(int, lambda x: x > 0),
            "clusters_distance_threshold": Or(None, float),
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
            "Estimated optimal tile size for small"
            "component removal: {} meters".format(tile_size)
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
        merged_point_cloud,
        orchestrator=None,
        depth_map_dir=None,
        point_cloud_dir=None,
        dump_dir=None,
        epsg=None,
    ):
        """
        Run PointCloudOutlierRemoval application.

        Creates a CarsDataset filled with new point cloud tiles.

        :param merged_point_cloud: merged point cloud. CarsDataset contains:

            - Z x W Delayed tiles. \
                Each tile will be a future pandas DataFrame containing:

                - data : with keys : "x", "y", "z", "corr_msk"\
                    optional: "clr", "msk", "data_valid","coord_epi_geom_i",\
                     "coord_epi_geom_j","idx_im_epi" \
                - attrs with keys: "epsg"
            - attributes containing "bounds", "ysize", "xsize", "epsg"

        :type merged_point_cloud: CarsDataset filled with pandas.DataFrame
        :param orchestrator: orchestrator used
        :param depth_map_dir: output depth map directory. If None output will
            be written in dump_dir if intermediate data is requested
        :type depth_map_dir: str
        :param point_cloud_dir: output depth map directory. If None output will
            be written in dump_dir if intermediate data is requested
        :type point_cloud_dir: str
        :type output_dir: str
        :param dump_dir: dump dir for output (except depth map) if intermediate
            data is requested
        :type dump_dir: str
        :param epsg: cartographic reference for the point cloud (array input)
        :type epsg: int

        :return: filtered merged point cloud. CarsDataset contains:

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
                saving_info,
            ) = self.__register_pc_dataset__(
                merged_point_cloud,
                point_cloud_dir,
                dump_dir,
                app_name="small_components",
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
                            small_component_removal_wrapper
                        )(
                            merged_point_cloud[row, col],
                            self.connection_distance,
                            self.nb_points_threshold,
                            self.clusters_distance_threshold,
                            save_by_pair=(self.save_by_pair),
                            point_cloud_csv_file_name=point_cloud_csv_file_name,
                            point_cloud_laz_file_name=point_cloud_laz_file_name,
                            saving_info=full_saving_info,
                        )

        elif merged_point_cloud.dataset_type == "arrays":
            prefix = os.path.basename(dump_dir)
            # Save as depth map
            filtered_point_cloud, saving_info_epipolar = (
                self.__register_epipolar_dataset__(
                    merged_point_cloud,
                    depth_map_dir,
                    dump_dir,
                    app_name="small_components",
                    pair_key=prefix,
                )
            )

            # Save as point cloud
            (
                flatten_filtered_point_cloud,
                laz_pc_dir_name,
                csv_pc_dir_name,
                saving_info_flatten,
            ) = self.__register_pc_dataset__(
                merged_point_cloud,
                point_cloud_dir,
                dump_dir,
                app_name="small_components",
            )

            # initialize empty index file for point cloud product if official
            # product is requested
            pc_index = None
            if point_cloud_dir:
                pc_index = {}

            # Generate rasters
            for col in range(filtered_point_cloud.shape[1]):
                for row in range(filtered_point_cloud.shape[0]):

                    # update saving infos  for potential replacement
                    full_saving_info_epipolar = ocht.update_saving_infos(
                        saving_info_epipolar, row=row, col=col
                    )
                    full_saving_info_flatten = None
                    if saving_info_flatten is not None:
                        full_saving_info_flatten = ocht.update_saving_infos(
                            saving_info_flatten, row=row, col=col
                        )
                    if merged_point_cloud[row][col] is not None:
                        csv_pc_file_name, laz_pc_file_name = (
                            generate_point_cloud_file_names(
                                csv_pc_dir_name,
                                laz_pc_dir_name,
                                row,
                                col,
                                pc_index,
                                pair_key=prefix,
                            )
                        )
                        window = merged_point_cloud.tiling_grid[row, col]
                        overlap = merged_point_cloud.overlaps[row, col]
                        # Delayed call to cloud filtering
                        (
                            filtered_point_cloud[row, col],
                            flatten_filtered_point_cloud[row, col],
                        ) = self.orchestrator.cluster.create_task(
                            epipolar_small_component_removal_wrapper, nout=2
                        )(
                            merged_point_cloud[row, col],
                            self.connection_distance,
                            self.nb_points_threshold,
                            self.clusters_distance_threshold,
                            self.half_epipolar_size,
                            window,
                            overlap,
                            epsg=epsg,
                            point_cloud_csv_file_name=csv_pc_file_name,
                            point_cloud_laz_file_name=laz_pc_file_name,
                            saving_info_epipolar=full_saving_info_epipolar,
                            saving_info_flatten=full_saving_info_flatten,
                        )

            # update point cloud index
            if point_cloud_dir:
                self.orchestrator.update_index(pc_index)

        else:
            logging.error(
                "PointCloudOutlierRemoval application doesn't support "
                "this input data "
                "format"
            )

        return filtered_point_cloud


def small_component_removal_wrapper(
    cloud,
    connection_distance,
    nb_points_threshold,
    clusters_distance_threshold,
    save_by_pair: bool = False,
    point_cloud_csv_file_name=None,
    point_cloud_laz_file_name=None,
    saving_info=None,
):
    """
    Small components outlier removal

    :param cloud: cloud to filter
    :type cloud: pandas DataFrame
    :param connection_distance: connection distance
    :type connection_distance: float
    :param nb_points_threshold:
    :type nb_points_threshold: int
    :param clusters_distance_threshold:
    :type clusters_distance_threshold: float
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
    cloud_attributes = cars_dataset.get_attributes(new_cloud)
    cloud_epsg = cloud_attributes["epsg"]
    current_epsg = cloud_epsg

    # Check if can be used to filter
    spatial_ref = CRS.from_epsg(cloud_epsg)
    if spatial_ref.is_geographic:
        logging.debug(
            "The points cloud to filter is not in a cartographic system. "
            "The filter's default parameters might not be adapted "
            "to this referential. Please, convert the points "
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
    (
        new_cloud,
        _,
    ) = outlier_removal_tools.small_component_filtering(
        new_cloud,
        connection_distance,
        nb_points_threshold,
        clusters_distance_threshold,
    )
    toc = time.process_time()
    logging.debug(
        "Small component cloud filtering done in {} seconds".format(toc - tic)
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


def epipolar_small_component_removal_wrapper(
    cloud,
    connection_distance,
    nb_points_threshold,
    clusters_distance_threshold,
    half_epipolar_size,
    window,
    overlap,
    epsg,
    point_cloud_csv_file_name=None,
    point_cloud_laz_file_name=None,
    saving_info_epipolar=None,
    saving_info_flatten=None,
):
    """
    Small component outlier removal in epipolar geometry

    :param epipolar_ds: epipolar dataset to filter
    :type epipolar_ds: xr.Dataset
    :param connection_distance: minimum distance of two connected points
    :type connection_distance: float
    :param nb_points_threshold: minimum valid cluster size
    :type nb_points_threshold: int
    :param clusters_distance_threshold: max distance between an outlier cluster
        and other points
    :type clusters_distance_threshold: float
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
    filtered_cloud = copy.copy(cloud)

    outlier_removal_tools.epipolar_small_components(
        filtered_cloud,
        epsg,
        min_cluster_size=nb_points_threshold,
        radius=connection_distance,
        half_window_size=half_epipolar_size,
        clusters_distance_threshold=clusters_distance_threshold,
    )

    # Fill with attributes
    cars_dataset.fill_dataset(
        filtered_cloud,
        saving_info=saving_info_epipolar,
        window=cars_dataset.window_array_to_dict(window),
        profile=None,
        attributes=None,
        overlaps=cars_dataset.overlap_array_to_dict(overlap),
    )

    # Flatten point cloud to save it as LAZ
    flatten_filtered_cloud = None
    if point_cloud_csv_file_name or point_cloud_laz_file_name:
        # Convert epipolar array into point cloud
        flatten_filtered_cloud, cloud_epsg = (
            point_cloud_tools.create_combined_cloud(
                [filtered_cloud], ["0"], epsg
            )
        )
        # Convert to UTM
        if epsg is not None and cloud_epsg != epsg:
            projection.point_cloud_conversion_dataframe(
                flatten_filtered_cloud, cloud_epsg, epsg
            )
            cloud_epsg = epsg

        # Fill attributes for LAZ saving
        color_type = point_cloud_tools.get_color_type([filtered_cloud])

        attributes = {
            "epsg": cloud_epsg,
            "color_type": color_type,
            cst.CROPPED_DISPARITY_RANGE: ocht.get_disparity_range_cropped(
                cloud
            ),
        }

        cars_dataset.fill_dataframe(
            flatten_filtered_cloud,
            saving_info=saving_info_flatten,
            attributes=attributes,
        )

    # Save point cloud in worker
    if point_cloud_csv_file_name:
        cars_dataset.run_save_points(
            flatten_filtered_cloud,
            point_cloud_csv_file_name,
            overwrite=True,
            point_cloud_format="csv",
            overwrite_file_name=False,
        )
    if point_cloud_laz_file_name:
        cars_dataset.run_save_points(
            flatten_filtered_cloud,
            point_cloud_laz_file_name,
            overwrite=True,
            point_cloud_format="laz",
            overwrite_file_name=False,
        )

    return filtered_cloud, flatten_filtered_cloud
