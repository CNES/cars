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
import os

import numpy as np

# Third party imports
from json_checker import And, Checker, Or
from pyproj import CRS

# CARS imports
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.point_cloud_outlier_removal import (
    abstract_outlier_removal_app as pc_removal,
)
from cars.applications.point_cloud_outlier_removal import (
    outlier_removal_algo,
)
from cars.applications.triangulation import pc_transform
from cars.applications.triangulation.triangulation_wrappers import (
    generate_point_cloud_file_names,
)
from cars.core import constants as cst
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

    def __init__(self, scaling_coeff, conf=None):
        """
        Init function of Statistical

        :param scaling_coeff: scaling factor for resolution
        :type scaling_coeff: float
        :param conf: configuration for points outlier removal
        :return: a application_to_use object
        """

        super().__init__(scaling_coeff, conf=conf)

        self.used_method = self.used_config["method"]

        # statistical outliers
        self.k = self.used_config["k"]
        self.filtering_constant = self.used_config["filtering_constant"]
        self.mean_factor = self.used_config["mean_factor"]
        self.std_dev_factor = self.used_config["std_dev_factor"]
        self.use_median = self.used_config["use_median"]
        self.half_epipolar_size = self.used_config["half_epipolar_size"]

        # Saving files
        self.save_intermediate_data = self.used_config["save_intermediate_data"]
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
        overloaded_conf["use_median"] = conf.get("use_median", True)

        # statistical outlier filtering
        # k: number of neighbors
        overloaded_conf["k"] = conf.get("k", 50)
        # filtering_constant: constant to apply in the distance threshold
        # computation
        overloaded_conf["filtering_constant"] = conf.get(
            "filtering_constant", 0.0
        )
        # mean_factor: factor to apply to the mean in the distance threshold
        # computation
        overloaded_conf["mean_factor"] = conf.get("mean_factor", 1.3)
        # mean_factor: factor to apply to the standard deviation in the
        # distance threshold
        overloaded_conf["std_dev_factor"] = conf.get("std_dev_factor", 3.0)

        # half_epipolar_size:
        # Half size of the epipolar window used for neighobr search (depth map
        # input only)
        overloaded_conf["half_epipolar_size"] = conf.get(
            "half_epipolar_size", 5
        )

        point_cloud_outlier_removal_schema = {
            "method": str,
            "k": And(int, lambda x: x > 0),
            "filtering_constant": And(Or(float, int), lambda x: x >= 0),
            "mean_factor": And(Or(float, int), lambda x: x >= 0),
            "std_dev_factor": And(Or(float, int), lambda x: x >= 0),
            "use_median": bool,
            "half_epipolar_size": int,
            application_constants.SAVE_INTERMEDIATE_DATA: bool,
        }

        # Check conf
        checker = Checker(point_cloud_outlier_removal_schema)
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

    def get_epipolar_margin(self):
        """
        Get epipolar margin to use

        :return: margin
        :rtype: int
        """
        margin = self.half_epipolar_size

        return margin

    def get_on_ground_margin(self, resolution=0.5):
        """
        Get margins to use during point clouds fusion

        :return: margin
        :rtype: float

        """

        return 0

    def run(  # pylint: disable=too-many-positional-arguments
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

                - data : with keys : "x", "y", "z", "corr_msk" \
                    optional: "clr", "msk", "data_valid","coord_epi_geom_i",\
                     "coord_epi_geom_j", "idx_im_epi"
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

        if dump_dir is None:
            dump_dir = self.generate_unknown_dump_dir(self.orchestrator)

        if merged_point_cloud.dataset_type != "arrays":
            raise RuntimeError(
                "Only arrays is supported in statistical removal"
            )

        prefix = os.path.basename(dump_dir)
        # Save as depth map
        filtered_point_cloud, saving_info_epipolar = (
            self.__register_epipolar_dataset__(
                merged_point_cloud,
                depth_map_dir,
                dump_dir,
                app_name="statistical",
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
            app_name="statistical",
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
                    overlap = filtered_point_cloud.overlaps[row, col]
                    # Delayed call to cloud filtering
                    (
                        filtered_point_cloud[row, col],
                        flatten_filtered_point_cloud[row, col],
                    ) = self.orchestrator.cluster.create_task(
                        epipolar_statistical_removal_wrapper, nout=2
                    )(
                        merged_point_cloud[row, col],
                        self.k,
                        self.filtering_constant,
                        self.mean_factor,
                        self.std_dev_factor,
                        self.use_median,
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

        return filtered_point_cloud


# pylint: disable=too-many-positional-arguments
def epipolar_statistical_removal_wrapper(
    epipolar_ds,
    statistical_k,
    filtering_constant,
    mean_factor,
    std_dev_factor,
    use_median,
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
    Statistical outlier removal in epipolar geometry

    :param epipolar_ds: epipolar dataset to filter
    :type epipolar_ds: xr.Dataset
    :param statistical_k: k
    :type statistical_k: int
    :param filtering_constant: constant applied to the threshold
    :type filtering_constant: float
    :param mean_factor: mean factor
    :type mean_factor: float
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

    # Get current epsg
    cloud_epsg = filtered_cloud.attrs["epsg"]
    current_epsg = cloud_epsg

    # Check if can be used to filter
    spatial_ref = CRS.from_epsg(cloud_epsg)
    if spatial_ref.is_geographic:
        logging.debug(
            "The point cloud to filter is not in a cartographic system. "
            "The filter's default parameters might not be adapted "
            "to this referential. Please, convert the point "
            "cloud to ECEF to ensure a proper point_cloud."
        )
        # Convert to epsg = 4978
        cartographic_epsg = 4978

        projection.point_cloud_conversion_dataset(
            filtered_cloud, cartographic_epsg
        )
        current_epsg = cartographic_epsg

    outlier_removal_algo.epipolar_statistical_filtering(
        filtered_cloud,
        k=statistical_k,
        filtering_constant=filtering_constant,
        mean_factor=mean_factor,
        dev_factor=std_dev_factor,
        use_median=use_median,
        half_window_size=half_epipolar_size,
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
            pc_transform.depth_map_dataset_to_dataframe(
                filtered_cloud, current_epsg
            )
        )
        # Convert to wanted epsg
        if epsg is not None and cloud_epsg != epsg:
            projection.point_cloud_conversion_dataframe(
                flatten_filtered_cloud, cloud_epsg, epsg
            )
            cloud_epsg = epsg

        # Fill attributes for LAZ saving
        color_type = pc_transform.get_color_type([filtered_cloud])
        attributes = {
            "epsg": cloud_epsg,
            "color_type": color_type,
            cst.CROPPED_DISPARITY_RANGE: ocht.get_disparity_range_cropped(
                epipolar_ds
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
