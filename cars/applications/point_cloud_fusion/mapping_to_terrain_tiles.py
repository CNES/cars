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
this module contains the epipolar cloud fusion application class.
"""


# Standard imports
import logging
import math
import os
from collections import Counter
from typing import List

# Third party imports
import numpy as np
import pandas
import xarray as xr
from json_checker import Checker, Or

# CARS imports
import cars.orchestrator.orchestrator as ocht
from cars.applications import application_constants
from cars.applications.point_cloud_fusion import (
    cloud_fusion_constants,
    point_cloud_tools,
)
from cars.applications.point_cloud_fusion.point_cloud_fusion import (
    PointCloudFusion,
)
from cars.core import tiling
from cars.data_structures import cars_dataset, format_transformation


class MappingToTerrainTiles(
    PointCloudFusion, short_name="mapping_to_terrain_tiles"
):
    """
    EpipolarCloudFusion
    """

    def __init__(self, conf=None):
        """
        Init function of EpipolarCloudFusion

        :param conf: configuration for fusion
        :return: an application_to_use object
        """

        super().__init__(conf=conf)

        # Cloud fusion
        self.used_method = self.used_config["method"]
        self.terrain_tile_size = self.used_config["terrain_tile_size"]
        self.resolution = self.used_config["resolution"]

        # check loader

        # Saving files
        self.save_points_cloud = self.used_config.get(
            "save_points_cloud", False
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
        overloaded_conf["method"] = conf.get(
            "method", "mapping_to_terrain_tiles"
        )
        overloaded_conf["terrain_tile_size"] = conf.get(
            "terrain_tile_size", None
        )
        overloaded_conf["resolution"] = conf.get("resolution", 0.5)
        overloaded_conf["save_points_cloud"] = conf.get(
            "save_points_cloud", False
        )

        points_cloud_fusion_schema = {
            "method": str,
            "terrain_tile_size": Or(int, None),
            "resolution": float,
            "save_points_cloud": bool,
        }

        # Check conf
        checker = Checker(points_cloud_fusion_schema)
        checker.validate(overloaded_conf)

        return overloaded_conf

    def pre_run(self, bounds, optimal_terrain_tile_width):
        """
        Pre run some computations


        :return: bounds, terrain_grid
        """

        # Compute optimal tile size

        if self.terrain_tile_size is None:
            # In case of multiple json configuration,
            # take the average optimal size,
            # and align to multiple of resolution

            optimal_terrain_tile_width = (
                int(math.ceil(optimal_terrain_tile_width / self.resolution))
                * self.resolution
            )
        else:
            optimal_terrain_tile_width = (
                self.terrain_tile_size * self.resolution
            )

        logging.info(
            "Optimal terrain tile size: {}x{} pixels".format(
                int(optimal_terrain_tile_width / self.resolution),
                int(optimal_terrain_tile_width / self.resolution),
            )
        )

        [xmin, ymin, xmax, ymax] = bounds

        # Split terrain bounding box in pieces
        terrain_grid = tiling.grid(
            xmin,
            ymin,
            xmax,
            ymax,
            optimal_terrain_tile_width,
            optimal_terrain_tile_width,
        )

        return terrain_grid

    def run(
        self,
        list_epipolar_points_cloud_left,
        list_epipolar_points_cloud_right,
        bounds,
        epsg,
        orchestrator=None,
        margins=None,
        on_ground_margin=0,
        optimal_terrain_tile_width=500,
    ):
        """
        Run EpipolarCloudFusion application.

        Creates a CarsDataset corresponding to the merged points clouds,
        tiled with the terrain grid used during rasterization.

        :param list_epipolar_points_cloud_left: list with left points clouds\
            Each CarsDataset contains:

            - N x M Delayed tiles. \
                Each tile will be a future xarray Dataset containing:

                - data : with keys : "x", "y", "z", "corr_msk" \
                    optional: "color", "msk",
                - attrs with keys: "margins", "epi_full_size", "epsg"
            - attributes containing: "disp_lower_bound",  "disp_upper_bound" \
                "elevation_delta_lower_bound","elevation_delta_upper_bound"
        :type list_epipolar_points_cloud_left: list(CarsDataset) filled with
          xr.Dataset
        :param list_epipolar_points_cloud_right: list with right points clouds.\
            Each CarsDataset contains:

            - N x M Delayed tiles.\
                Each tile will be a future xarray Dataset containing:

                - data : with keys : "x", "y", "z", "corr_msk" \
                    optional: "color", "msk",
                - attrs with keys: "margins", "epi_full_size", "epsg"
            - attributes containing: "disp_lower_bound",  "disp_upper_bound",\
                "elevation_delta_lower_bound","elevation_delta_upper_bound"
        :type list_epipolar_points_cloud_right: list(CarsDataset) filled with
          xr.Dataset
        :param bounds: terrain bounds
        :type bounds: list
        :param epsg: epsg to use
        :type epsg: str
        :param orchestrator: orchestrator used
        :type orchestrator: Orchestrator
        :param margins: margins to add to tiles
        :type margins: dict
            ex: {"radius": 1, "resolution": 0.5}
        :param on_ground_margin: margins needed for future filtering
        :type on_ground_margin: float
        :param optimal_terrain_tile_width: optimal terrain tile width
        :type optimal_terrain_tile_width: int

        :return: Merged points clouds

            CarsDataset contains:

            - Z x W Delayed tiles\
                Each tile will be a future pandas DataFrame containing:

                - data : with keys : "x", "y", "z", "corr_msk" \
                    optional: "clr", "msk", "data_valid","coord_epi_geom_i",\
                     "coord_epi_geom_j","idx_im_epi"
                - attrs with keys: "epsg"
            - attributes containing: "bounds", "ysize", "xsize", "epsg"

        :rtype: CarsDataset filled with pandas.DataFrame

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

        if margins is None:
            margins = {"radius": None, "resolution": None}

        # Compute bounds and terrain grid
        terrain_grid = self.pre_run(bounds, optimal_terrain_tile_width)

        if list_epipolar_points_cloud_left[0].dataset_type == "arrays":

            # Derive output image files parameters to pass to rasterio
            xsize, ysize = tiling.roi_to_start_and_size(
                [bounds[0], bounds[1], bounds[2], bounds[3]], self.resolution
            )[2:]
            logging.info(
                "DSM output image size: {}x{} pixels".format(xsize, ysize)
            )

            # Create CarsDataset
            merged_point_cloud = cars_dataset.CarsDataset("points")

            # Compute tiling grid
            merged_point_cloud.tiling_grid = (
                format_transformation.tiling_grid_2_cars_dataset_grid(
                    terrain_grid, resolution=self.resolution, from_terrain=True
                )
            )

            # update attributes
            merged_point_cloud.attributes["bounds"] = bounds
            merged_point_cloud.attributes["ysize"] = ysize
            merged_point_cloud.attributes["xsize"] = xsize
            merged_point_cloud.attributes["epsg"] = epsg

            # Save objects

            if self.save_points_cloud:
                # Points cloud file name
                # TODO in input conf file
                pc_file_name = os.path.join(
                    self.orchestrator.out_dir, "points_cloud.csv"
                )
                self.orchestrator.add_to_save_lists(
                    pc_file_name,
                    None,
                    merged_point_cloud,
                    cars_ds_name="merged_points_cloud",
                )

            # Get saving infos in order to save tiles when they are computed
            [saving_info] = self.orchestrator.get_saving_infos(
                [merged_point_cloud]
            )

            number_of_terrain_tiles = (
                merged_point_cloud.tiling_grid.shape[1]
                * merged_point_cloud.tiling_grid.shape[0]
            )

            logging.info(
                "Number of tiles in cloud fusion :"
                "row : {} "
                "col : {}".format(
                    merged_point_cloud.shape[0],
                    merged_point_cloud.shape[1],
                )
            )

            number_of_epipolar_tiles_per_terrain_tiles = []

            # Add epipolar_points_min and epipolar_points_max used
            #  in point_cloud_fusion
            # , to get corresponding tiles (terrain)
            # TODO change method for corresponding tiles
            list_points_min = []
            list_points_max = []
            for pc_left in list_epipolar_points_cloud_left:
                points_min, points_max = tiling.terrain_grid_to_epipolar(
                    terrain_grid,
                    pc_left.attributes["epipolar_regions_grid"],
                    pc_left.attributes["epipolar_grid_min"],
                    pc_left.attributes["epipolar_grid_max"],
                    epsg,
                )
                list_points_min.append(points_min)
                list_points_max.append(points_max)

            # Add infos to orchestrator.out_json
            updating_dict = {
                application_constants.APPLICATION_TAG: {
                    cloud_fusion_constants.CLOUD_FUSION_RUN_TAG: {
                        cloud_fusion_constants.EPSG_TAG: epsg,
                        cloud_fusion_constants.MARGINS_TAG: margins,
                        cloud_fusion_constants.ON_GROUND_MARGINS_TAG: (
                            on_ground_margin
                        ),
                        cloud_fusion_constants.NUMBER_TERRAIN_TILES: (
                            number_of_terrain_tiles
                        ),
                        cloud_fusion_constants.XSIZE: xsize,
                        cloud_fusion_constants.YSIZE: ysize,
                        cloud_fusion_constants.BOUNDS: bounds,
                    },
                }
            }
            orchestrator.update_out_info(updating_dict)

            # Generate merged point clouds
            logging.info(
                "Point clouds: Merged points number: {}".format(
                    merged_point_cloud.shape[1] * merged_point_cloud.shape[0]
                )
            )

            for col in range(merged_point_cloud.shape[1]):
                for row in range(merged_point_cloud.shape[0]):

                    # Get required point clouds
                    (
                        terrain_region,
                        required_point_clouds_left,
                        required_point_clouds_right,
                        _rank,
                        _pos,
                    ) = tiling.get_corresponding_tiles_row_col(
                        terrain_grid,
                        row,
                        col,
                        list_epipolar_points_cloud_left,
                        list_epipolar_points_cloud_right,
                        list_points_min,
                        list_points_max,
                    )

                    if len(required_point_clouds_left) > 0:
                        logging.debug(
                            "Number of clouds to process for this terrain"
                            " tile: {}".format(len(required_point_clouds_left))
                        )
                        number_of_epipolar_tiles_per_terrain_tiles.append(
                            len(required_point_clouds_left)
                        )

                        # start and size parameters for the rasterization
                        # function
                        (
                            xstart,
                            ystart,
                            xsize,
                            ysize,
                        ) = tiling.roi_to_start_and_size(
                            terrain_region, self.resolution
                        )

                        # Delayed call to rasterization operations using all
                        #  required point clouds
                        merged_point_cloud[
                            row, col
                        ] = self.orchestrator.cluster.create_task(
                            compute_point_cloud_wrapper
                        )(
                            required_point_clouds_left,
                            required_point_clouds_right,
                            margins["resolution"],
                            epsg,
                            xstart=xstart,
                            ystart=ystart,
                            xsize=xsize,
                            ysize=ysize,
                            radius=margins["radius"],
                            on_ground_margin=on_ground_margin,
                            saving_info=saving_info,
                        )

            # Sort tiles according to rank TODO remove or implement it ?

            # Add delayed_dsm_tiles to orchestrator
            logging.info(
                "Submitting {} tasks to dask".format(number_of_terrain_tiles)
            )

            logging.info(
                "Number of epipolar tiles "
                "for each terrain tile (counter): {}".format(
                    sorted(
                        Counter(
                            number_of_epipolar_tiles_per_terrain_tiles
                        ).items()
                    )
                )
            )

            logging.info(
                "Average number of epipolar tiles "
                "for each terrain tile: {}".format(
                    int(
                        np.round(
                            np.mean(number_of_epipolar_tiles_per_terrain_tiles)
                        )
                    )
                )
            )

            logging.info(
                "Max number of epipolar tiles "
                "for each terrain tile: {}".format(
                    np.max(number_of_epipolar_tiles_per_terrain_tiles)
                )
            )

        else:
            logging.error(
                "PointsCloudRasterisation application doesn't "
                "support this input data format"
            )

        return merged_point_cloud


def compute_point_cloud_wrapper(
    point_clouds_left, point_clouds_right, resolution, epsg, **kwargs
):
    """
    Wrapper for points clouds fusion step :
    - Convert a list of clouds to correct epsg

    :param point_clouds_left: list of clouds, list of datasets with :

            - cst.X
            - cst.Y
            - cst.Z
            - cst.EPI_COLOR
    :type point_clouds_left: list(xr.Dataset)
    :param point_clouds_right: list of cloud, list of datasets \
           (list of None if use_sec_disp not activated) with :

            - cst.X
            - cst.Y
            - cst.Z
            - cst.EPI_COLOR
    :type point_clouds_right: list of DataObject
    :param resolution: Produced DSM resolution (meter, degree [EPSG dependent])
    :type resolution: float
    :param  epsg_code: epsg code for the CRS of the output DSM
    :type epsg_code: int
    :param  stereo_out_epsg: epsg code to convert point cloud to, if needed
    :type stereo_out_epsg: int

    :return: merged points cloud dataframe with:
            - cst.X
            - cst.Y
            - cst.Z
            - cst.EPI_COLOR
            - attrs : xstart, ystart, xsize, ysize, saving_info
    :rtype: pandas.DataFrame
    """
    # Unpack list of clouds from tuple, and project them to correct EPSG if
    # needed
    clouds = point_clouds_left

    # Add clouds and colors computed from the secondary disparity map
    if point_clouds_right[0] is not None:
        cloud_sec = point_clouds_right
        clouds.extend(cloud_sec)

    # Call simple_rasterization
    pc_pandas, cloud_epsg = simple_merged_point_cloud_dataset(
        clouds,
        resolution,
        epsg,
        xstart=kwargs["xstart"],
        ystart=kwargs["ystart"],
        xsize=kwargs["xsize"],
        ysize=kwargs["ysize"],
        radius=kwargs["radius"],
        on_ground_margin=kwargs["on_ground_margin"],
    )

    # Fill attributes for rasterization
    attributes = {
        "xstart": kwargs["xstart"],
        "ystart": kwargs["ystart"],
        "xsize": kwargs["xsize"],
        "ysize": kwargs["ysize"],
        "epsg": cloud_epsg,
    }
    cars_dataset.fill_dataframe(
        pc_pandas, saving_info=kwargs["saving_info"], attributes=attributes
    )

    return pc_pandas


def simple_merged_point_cloud_dataset(
    cloud_list: List[xr.Dataset],
    resolution: float,
    epsg: int,
    xstart: float = None,
    ystart: float = None,
    xsize: int = None,
    ysize: int = None,
    radius: int = 1,
    on_ground_margin=0,
) -> pandas.DataFrame:
    """
    Wrapper of simple_rasterization
    that has xarray.Dataset as inputs and outputs.

    :param cloud_list: list of cloud points to rasterize
    :param resolution: Resolution of rasterized cells,
        expressed in cloud CRS units or None
    :param epsg: epsg code for the CRS of the final raster
    :param xstart: xstart of the rasterization grid
        (if None, will be estimated by the function)
    :param ystart: ystart of the rasterization grid
        (if None, will be estimated by the function)
    :param xsize: xsize of the rasterization grid
        (if None, will be estimated by the function)
    :param ysize: ysize of the rasterization grid
        (if None, will be estimated by the function)
    :param radius: rasterization radius
    :type radius: int
    :param on_ground_margin: point cloud filtering margin
    :type on_ground_margin: int

    :return: cloud and Color
    """

    # combined clouds
    cloud, cloud_epsg = point_cloud_tools.create_combined_cloud(
        cloud_list,
        epsg,
        resolution=resolution,
        xstart=xstart,
        ystart=ystart,
        xsize=xsize,
        ysize=ysize,
        on_ground_margin=on_ground_margin,
        epipolar_border_margin=0,
        radius=radius,
        with_coords=True,
    )

    return cloud, cloud_epsg
